"""AWS 밖 후보(Cloud Run, Azure ML, Cloudflare, 무료 호스팅)를 조건표로 거른다.

candidates.json의 값은 공식 문서를 2026-11-14에 열어 확인한 학습용 스냅샷이다.
가격은 넣지 않는다. 승자를 정하지 않고 BLOCKED / CANDIDATE / CANDIDATE_WITH_UNKNOWNS로만 나눈다.
약관은 코드가 판단하지 못하므로 모든 후보에 직접 확인할 항목으로 남긴다.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent

PASS, BLOCK, UNKNOWN = "PASS", "BLOCK", "UNKNOWN"


def load_candidates(path: Path | None = None) -> dict[str, Any]:
    path = path or HERE / "candidates.json"
    return json.loads(path.read_text(encoding="utf-8"))["candidates"]


# Cloudflare R2 단일 업로드 한도(GiB). https://developers.cloudflare.com/r2/platform/limits/ 2026-11-14 확인.
# 한 번에 올리는 단일 요청 한도는 표에 5 GiB로 적혀 있다. 나눠 올리는 멀티파트 한도는 표(4.995 TiB)와 각주(4.995 GiB)의 단위가 어긋나지만, 이 코드는 단일 업로드 한도만 쓰므로 영향이 없다.
R2_SINGLE_UPLOAD_GIB = 5


def gb_to_gib(gb: float) -> float:
    """십진 GB를 이진 GiB로 바꾼다(1GB = 10^9바이트, 1GiB = 2^30바이트)."""
    return gb * 10**9 / 2**30


def fits_single_upload(file_gb: float, limit_gib: float = R2_SINGLE_UPLOAD_GIB) -> bool:
    """모델 파일이 한 번에 올리는 업로드 한도 안에 드는가. 넘으면 나눠 올리는 멀티파트가 필요하다."""
    return gb_to_gib(file_gb) <= limit_gib


def peak_rpm(profile: dict[str, Any]) -> float:
    """몰리는 구간의 분당 요청 수. 하루 요청 중 burst_share가 burst_window_seconds 안에 온다고 가정한다."""
    burst_requests = profile["requests_per_day"] * profile["burst_share"]
    return burst_requests / (profile["burst_window_seconds"] / 60)


def _val(fields: dict[str, Any], name: str) -> Any:
    return fields.get(name, {}).get("value")


def run_checks(profile: dict[str, Any], fields: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(검사 이름, PASS/BLOCK/UNKNOWN, 이유) 목록을 돌려준다."""
    out: list[tuple[str, str, str]] = []

    # 1. GPU와 내 모델 파일
    gpu = _val(fields, "gpu")
    if profile["gpu_required"]:
        if gpu == "selectable":
            out.append(("gpu", PASS, "GPU를 직접 고를 수 있음"))
        elif gpu == "none":
            out.append(("gpu", BLOCK, "GPU 미지원"))
        elif gpu == "managed_catalog":
            out.append(("gpu", UNKNOWN, "제공 모델 목록 기준 서비스라 내 GPU 모델을 올릴 수 있는지 문서로 확인 필요"))
        else:
            out.append(("gpu", UNKNOWN, "GPU 지원 여부를 문서에서 확인하지 못함"))

    weights = _val(fields, "own_full_weights")
    if profile["own_full_weights"]:
        if weights == "container":
            out.append(("own_weights", PASS, "컨테이너로 내 모델을 올림"))
        elif weights == "lora_only_documented":
            out.append(("own_weights", UNKNOWN, "문서에는 LoRA 어댑터만 적혀 있어 전체 모델 가능 여부 확인 필요"))
        else:
            out.append(("own_weights", UNKNOWN, "내 모델을 올리는 방식을 문서에서 확인하지 못함"))

    # 2. 메모리
    memory = _val(fields, "max_memory_gb")
    need = profile["runtime_memory_gb"]
    if memory is None:
        out.append(("memory", UNKNOWN, "메모리 상한을 확인하지 못함"))
    elif memory < need:
        out.append(("memory", BLOCK, f"메모리 상한 {memory}GB가 필요량 {need}GB보다 작음"))
    else:
        out.append(("memory", PASS, f"메모리 상한 {memory}GB가 필요량 {need}GB 이상"))

    # 3. 요청 한 건의 처리 시간
    limit = _val(fields, "max_request_seconds")
    background = _val(fields, "max_background_seconds")
    sec = profile["seconds_per_request"]
    if limit is not None and sec <= limit:
        out.append(("request_time", PASS, f"건당 {sec}초가 제한 {limit}초 이내"))
    elif background is not None and profile["async_allowed"] and sec <= background:
        out.append(("request_time", PASS, f"백그라운드 {background}초 이내(나중에 답을 받는 흐름)"))
    elif limit is not None:
        out.append(("request_time", BLOCK, f"건당 {sec}초가 제한 {limit}초를 넘음"))
    elif background is not None:
        out.append(("request_time", BLOCK, f"건당 {sec}초가 백그라운드 제한 {background}초를 넘거나 비동기가 허용되지 않음"))
    else:
        out.append(("request_time", UNKNOWN, "처리 시간 제한을 확인하지 못함"))

    # 4. 시작 지연 + 모델 불러오기
    cold = _val(fields, "cold_start_seconds")
    if cold is None:
        out.append(("startup", UNKNOWN, "시작 지연을 확인하지 못함"))
    else:
        total = cold + profile["model_load_seconds"]
        if total > profile["acceptable_wait_seconds"]:
            out.append(("startup", BLOCK, f"시작 {cold}초 + 모델 불러오기 {profile['model_load_seconds']}초 = {total}초가 허용 대기 {profile['acceptable_wait_seconds']}초를 넘음"))
        else:
            out.append(("startup", PASS, f"시작 {cold}초 + 모델 불러오기 {profile['model_load_seconds']}초 = {total}초"))

    # 5. 무료만 쓸 수 있는가
    if profile["free_only"]:
        tier = _val(fields, "free_tier")
        if tier == "documented":
            out.append(("free_tier", PASS, "무료 사용량이 문서에 정해져 있음"))
        elif tier == "paid_for_container":
            out.append(("free_tier", BLOCK, "컨테이너를 돌리는 Space는 유료 플랜이 필요함"))
        else:
            out.append(("free_tier", UNKNOWN, "무료 사용 범위를 확인하지 못함"))

    # 6. 분당 요청 한도(값이 있는 후보만)
    rpm = _val(fields, "rate_limit_rpm")
    if rpm is not None:
        peak = peak_rpm(profile)
        if peak <= rpm["min"]:
            out.append(("rate_limit", PASS, f"몰리는 구간 {peak:.2f}건/분이 하한 {rpm['min']}건/분 이내"))
        elif peak > rpm["max"]:
            out.append(("rate_limit", BLOCK, f"몰리는 구간 {peak:.2f}건/분이 상한 {rpm['max']}건/분을 넘음"))
        else:
            out.append(("rate_limit", UNKNOWN, f"몰리는 구간 {peak:.2f}건/분이 하한 {rpm['min']}과 상한 {rpm['max']} 사이라 내 모델의 분류를 확인해야 함"))
    return out


def evaluate(profile: dict[str, Any], candidates: dict[str, Any] | None = None) -> dict[str, Any]:
    candidates = candidates or load_candidates()
    result: dict[str, Any] = {"peak_rpm": round(peak_rpm(profile), 3), "candidates": {}}
    for cid, cand in candidates.items():
        checks = run_checks(profile, cand["fields"])
        states = [s for _, s, _ in checks]
        if BLOCK in states:
            status = "BLOCKED"
        elif UNKNOWN in states:
            status = "CANDIDATE_WITH_UNKNOWNS"
        else:
            status = "CANDIDATE"
        result["candidates"][cid] = {
            "name": cand["name"],
            "status": status,
            "checks": [{"check": c, "state": s, "reason": r} for c, s, r in checks],
            "caveats": cand.get("caveats", []),
            "manual": ["약관 원문(상업 이용·콘텐츠 제한)", "배포할 지역", "실제 단가와 최소 과금 단위"],
        }
    return result


def price_input_sheet(profile: dict[str, Any], candidates: dict[str, Any] | None = None) -> dict[str, Any]:
    """가격 대신 가격표에 넣을 '양'만 만든다. 단가는 null로 두고 사람이 공식 가격 페이지에서 채운다."""
    candidates = candidates or load_candidates()
    monthly_requests = profile["requests_per_day"] * profile["days_per_month"]
    active_seconds = monthly_requests * profile["seconds_per_request"]
    quantities = {
        "monthly_requests": monthly_requests,
        "active_seconds": active_seconds,
        "active_hours": active_seconds / 3600,
        "always_on_hours": profile["days_per_month"] * 24,
    }
    return {
        "quantities": quantities,
        "per_candidate": {cid: {"unit_price": None, "currency": None, "pricing_url": None, "checked_at": None} for cid in candidates},
    }


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "profile.json")
    profile = json.loads(path.read_text(encoding="utf-8"))
    result = evaluate(profile)
    print(f"몰리는 구간 분당 요청: {result['peak_rpm']}")
    for cid, item in result["candidates"].items():
        print(f"{item['status']:<26} {item['name']}")
        for c in item["checks"]:
            print(f"    [{c['state']:<7}] {c['check']}: {c['reason']}")
        for note in item["caveats"]:
            print(f"    [주의   ] {note}")
    print(f"모델 파일 4.683GB = {gb_to_gib(4.683):.3f}GiB, 한 번에 업로드 가능: {fits_single_upload(4.683)}")
    sheet = price_input_sheet(profile)
    print("가격표 입력량:", json.dumps(sheet["quantities"], ensure_ascii=False))


if __name__ == "__main__":
    main()
