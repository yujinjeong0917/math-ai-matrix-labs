# 2장 · 파일과 모델은 어디에 둘까

<!-- sync:start -->
> [AWS 배포·운영](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/aws) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [모델 파일과 실행 환경은 어디에 두고 어떻게 버전을 맞출까요?](https://math-ai-matrix-pi.vercel.app/tracks/aws/02-artifacts-s3-ecr.html)

```bash
cd aws/ch02_artifact_versions
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `artifact_rules.py` | 모델 파일(S3)과 컨테이너 이미지(ECR)의 버전 기록이 같은 릴리스를 가리키는지 검사한다. |
<!-- sync:end -->

릴리스 기록(`release_manifest.json`)이 모델 파일(S3)과 컨테이너 이미지(ECR)를 같은 버전으로 가리키는지 규칙 7개로 검사하고, S3에 버전을 쌓을 때의 저장량과 멀티파트 업로드 파트 수를 계산합니다. AWS 계정은 쓰지 않아요.

## 실행

```bash
python3 artifact_rules.py release_manifest.json
python3 -m unittest -q
```

## 기대 출력

```text
R3 FAIL: 이미지를 digest(@sha256:...)로 고정하지 않았어요
R4 FAIL: latest 태그는 나중에 다른 이미지를 가리킬 수 있어요
R5 WARN: ECR 태그 덮어쓰기를 막지 않았어요
R6 FAIL: S3 버킷 버전 관리가 켜져 있지 않아요
R7 FAIL: 서빙 코드 커밋이 기록돼 있지 않아요
판정: BLOCK
버전 10개 보관 시 저장량: 46.83GB
100 MiB 파트: {'parts': 45, 'fits': True}
파트 최소 크기: 5242880바이트 = 5 MiB, 파트 수 894
```

판정 기준: FAIL(탈락)이 하나라도 있으면 BLOCK, FAIL 없이 WARN(경고)만 있으면 WARN, 둘 다 없으면 PASS. R5(태그 덮어쓰기 허용)만 WARN인 것은 digest로 고정한 이미지(R3 통과)라면 태그가 바뀌어도 이미지가 바뀌지 않는다고 본 이 실습의 선택입니다. 표본 릴리스 기록은 R3이 FAIL이라 그 안전장치가 작동하지 않으므로, R5의 WARN은 R3을 고치기 전까지 덮어쓰기 위험이 남는다는 뜻으로 읽어야 해요.

## 확인하지 못하는 것

- 규칙 7개는 학습용입니다. 실제 보관 기간·삭제 정책(수명 주기)은 팀 규칙으로 따로 정해야 해요.
- digest 형식(`@sha256:` 뒤 64자리)은 이 실습이 만든 검사 규칙이며, ECR이 digest를 어떻게 계산하는지는 확인하지 않았습니다.
- `tag_mutability`가 비어 있을 때 `MUTABLE`로 보는 것은 이 코드의 선택이지 ECR 기본값 확인이 아니에요.
- 저장 비용은 계산하지 않아요. 저장량(GB)까지만 구합니다.
