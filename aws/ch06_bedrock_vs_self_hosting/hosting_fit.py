"""Bedrock Custom Model Import(직접 가져오기)가 내 모델 조건을 받아 주는지 가리는 검사기와,
토큰 쿼터·5분 청구 창 계산기. AWS를 호출하지 않고 가격을 넣지 않는다.

한도는 2026-11-14에 연 공식 문서 값이다.
  https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html
  https://docs.aws.amazon.com/bedrock/latest/userguide/invoke-imported-model.html
  https://docs.aws.amazon.com/bedrock/latest/userguide/import-model-calculate-cost.html
  https://docs.aws.amazon.com/bedrock/latest/userguide/quotas-token-burndown.html

재는 것: (1) 모델 프로필이 Custom Model Import 조건(구조, 형식, 크기, 문맥, 리전, API)을 통과하는지,
(2) 토큰 쿼터 차감량과 동시 요청 수, (3) 5분 청구 창 수.
입력: model_profile.json. 출력: cmi_possible과 막는 이유(README: 3개, GGUF 형식, 서울 리전, Converse 미지원).
아래 상수는 문서 확인값이며 바뀔 수 있다.
"""
import json
import math
import sys

CHECKED_AT = "2026-11-14"
CMI_REGIONS = {"eu-central-1", "us-east-1", "us-east-2", "us-west-2"}
GPT_OSS_ONLY_REGION = "us-east-1"
ARCHITECTURES = {"mistral", "mixtral", "flan", "llama", "gptbigcode", "qwen2", "qwen2.5", "qwen2-vl",
                 "qwen2.5-vl", "qwen3", "gpt-oss"}
TEXT_WEIGHTS_LIMIT_GB = 200      # 이 값 미만
MULTIMODAL_WEIGHTS_LIMIT_GB = 100
CONTEXT_LIMIT_K = 128            # "128K 미만". K가 1000인지 1024인지는 문서에 없다.
NO_CONVERSE = {"qwen2.5", "qwen3", "qwen2-vl", "qwen2.5-vl", "gpt-oss"}
WINDOW_SECONDS = 300             # 청구는 5분 창


def check_cmi(p):
    """(막는 이유, 확인할 점) 두 목록을 돌려준다. p는 model_profile.json 형식의 사전."""
    blockers, notes = [], []
    arch = p["architecture"].lower()
    if arch not in ARCHITECTURES:
        blockers.append(f"지원 구조 목록에 없음: {p['architecture']}")
    if p["weights_format"].lower() != "safetensors":
        blockers.append(f"가중치 형식이 {p['weights_format']}임. 문서는 Hugging Face 형식의 .safetensors 파일을 요구함")
    # 멀티모달(이미지 포함) 구조는 가중치 한도가 100GB로 더 작다. 이름이 -vl로 끝나거나 mllama
    multimodal = arch.endswith("-vl") or arch == "mllama"
    limit = MULTIMODAL_WEIGHTS_LIMIT_GB if multimodal else TEXT_WEIGHTS_LIMIT_GB
    # "미만" 조건이라 200GB 정확히도 탈락(>=)
    if p["weights_gb"] >= limit:
        blockers.append(f"가중치 {p['weights_gb']}GB가 한도({limit}GB 미만)를 넘음")
    ctx = p["max_context_tokens"]
    # K가 1024면 131072 이상은 확실히 탈락, 1000이면 128000부터라 그 사이(128000~131071)는 경고만
    if ctx >= CONTEXT_LIMIT_K * 1024:
        blockers.append(f"최대 문맥 {ctx}토큰이 128K 한도를 넘음")
    elif ctx >= CONTEXT_LIMIT_K * 1000:
        notes.append("문맥 길이가 128K 경계 근처라 K의 정의 확인 필요")
    if p["region"] not in CMI_REGIONS:
        blockers.append(f"리전 {p['region']}은 지원 목록({', '.join(sorted(CMI_REGIONS))})에 없음")
    # 필요하다고 한 API를 구조가 지원하지 않을 때만 탈락(Llama, Qwen2는 통과, Qwen2.5는 탈락)
    if p["needs_converse_api"] and arch in NO_CONVERSE:
        blockers.append(f"{p['architecture']} 구조는 Converse API 미지원(InvokeModel만 가능)")
    if p["needs_batch_inference"]:
        blockers.append("Custom Model Import는 Bedrock 배치 추론과 함께 쓸 수 없음")
    if p["needs_cloudformation"]:
        blockers.append("Custom Model Import는 CloudFormation과 함께 쓸 수 없음")
    if arch == "gpt-oss" and p["region"] != GPT_OSS_ONLY_REGION:
        blockers.append("GPT-OSS 구조는 us-east-1에서만 지원")
    notes.append("콜드 스타트: 비활성 모델은 내려가며, 다시 부르면 5분 안에 처리되거나 ModelNotReadyException이 날 수 있음")
    return blockers, notes


def quota_reserve(input_tokens, max_tokens):
    """요청이 시작될 때 쿼터에서 먼저 빼는 양."""
    # 실제 출력이 짧아도 시작 시점에는 최대치를 잡는다: 800 + 4,096 = 4,896
    return input_tokens + max_tokens


def quota_final(input_tokens, output_tokens, burndown=1):
    """요청이 끝난 뒤 실제로 쓴 것으로 정산되는 양(캐시 토큰 제외)."""
    # burndown은 출력 토큰의 소각률(모델마다 다름). 문서 예시 1,000 + 100 x 5 = 1,500
    return input_tokens + output_tokens * burndown


def concurrent_by_tpm(tpm_quota, input_tokens, max_tokens):
    """분당 토큰 쿼터 안에서 같은 순간에 시작할 수 있는 요청 수(시작 시점 차감 기준)."""
    # 쿼터 100,000(가상) // 4,896 = 20건, max_tokens를 512로 줄이면 // 1,312 = 76건
    return tpm_quota // quota_reserve(input_tokens, max_tokens)


def billing_windows(call_times_s):
    """첫 성공 호출부터 5분 창으로 나눴을 때 호출이 한 번이라도 있는 창의 수.
    창 안에서 모델이 얼마나 오래 유지되는지는 문서에 없어, 이 값은 하한 추정이다."""
    if not call_times_s:
        return 0
    t0 = min(call_times_s)  # 창의 시작은 첫 호출(문서의 "첫 성공 호출"과 같다고 가정)
    # 창 번호가 같은 호출은 한 창으로 센다: 0, 10, 299초는 1창, 0과 300초는 2창(경계는 다음 창)
    return len({int((t - t0) // WINDOW_SECONDS) for t in call_times_s})


def cmu_minutes(copies, cmu_per_copy, windows):
    """공식 식의 단가 앞부분: 복사본 x CMU x (창 수 x 5분). 여기에 단가를 곱하면 비용이 된다."""
    # 예: 복사본 1 x CMU 4(가상) x 창 20 x 5분 = 400 CMU·분 (README)
    return copies * cmu_per_copy * windows * 5


def run(profile):
    blockers, notes = check_cmi(profile)
    return {"checked_at": CHECKED_AT, "cmi_possible": not blockers, "blockers": blockers, "notes": notes}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "model_profile.json"
    with open(path, encoding="utf-8") as f:
        print(json.dumps(run(json.load(f)), ensure_ascii=False, indent=2))
