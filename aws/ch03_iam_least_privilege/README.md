# 3장 · 누가 무엇을 할 수 있나

<!-- sync:start -->
> [AWS 배포·운영](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/aws) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [누가 무엇을 할 수 있어야 할까요?](https://math-ai-matrix-pi.vercel.app/tracks/aws/03-iam-least-privilege.html)

```bash
cd aws/ch03_iam_least_privilege
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `policy_scan.py` | IAM 정책 JSON을 읽어 과한 권한을 찾는다. AWS 호출 없음, 표준 라이브러리만 쓴다. |
<!-- sync:end -->

IAM 정책 JSON을 읽어 `"*"` 권한, 서비스 전체 권한(`s3:*`), 모든 리소스(`"*"`), 어떤 역할이든 넘기는 `iam:PassRole`을 찾습니다. 또 "기본 거부, 명시적 Deny 우선" 규칙으로 요청 하나가 허용되는지 계산해요. AWS 계정은 쓰지 않아요.

## 실행

```bash
python3 policy_scan.py policies.json
python3 -m unittest -q
```

## 기대 출력

```text
[too_broad] HIGH 2, REVIEW 1, 점수 7
[half_broad] HIGH 1, REVIEW 5, 점수 8
[scoped] HIGH 0, REVIEW 0, 점수 0
scoped s3:GetObject example-model-bucket/reply-draft/v1/model.gguf -> ALLOW
scoped s3:GetObject example-model-bucket/other/secret.txt -> DENY(기본)
scoped s3:DeleteObject example-model-bucket/reply-draft/v1/model.gguf -> DENY(명시)
```

## 확인하지 못하는 것

- 이 검사기는 학습용 규칙입니다. 실제 정책은 IAM Access Analyzer의 정책 검증으로 다시 확인하세요.
- `Condition`, 권한 경계, 조직 정책(SCP·RCP), 리소스 기반 정책의 합집합·교집합은 계산하지 않아요. `allowed()`는 한 정책 안의 Allow·Deny만 봅니다.
- `NotAction`, `NotResource`, 정책 변수(`${aws:username}`)는 다루지 않아요.
- 점수는 문장을 세어 더한 값이라 정책끼리 줄 세우는 데 쓰지 않습니다. HIGH 개수를 먼저 보세요.
