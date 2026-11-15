# 12장 · 고객이 쓴 문장을 로그에 얼마나 남겨도 될까요?

<!-- sync:start -->
> [AWS 배포·운영](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/aws) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [고객이 쓴 문장을 로그에 얼마나 남겨도 될까요?](https://math-ai-matrix-pi.vercel.app/tracks/aws/12-prompt-log-privacy.html)

```bash
cd aws/ch12_prompt_log_privacy
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `log_policy.py` | 프롬프트 로그의 마스킹과 보관 기간을 AWS 없이 점검한다. |
<!-- sync:end -->

정규식 4개로 만든 학습용 마스킹, CloudWatch Logs 보관 기간 허용 값 검사, 보관 기간별 저장량·최대 잔존 일수 계산, 로그 그룹 설정 점검을 AWS 없이 돌립니다.

## 실행

```bash
python3 log_policy.py log_config.json
python3 -m unittest -v
```

## 기대 출력

```
마스킹 후: 문의: 홍길동([전화], [이메일]) 카드 [카드번호형] 환불해 주세요 {'카드번호형': 1, '전화': 1, '이메일': 1}
원문 로그 보관 7일: 최대 10일까지 남을 수 있음, 저장 2.8GB
원문 로그 보관 90일: 최대 93일까지 남을 수 있음, 저장 36.0GB
10%만 남기면 7일 저장 0.28GB
정책 3일 전에 켰다면 마스킹 안 된 건수 60000건
점검 기준: 원문 그룹의 설정 보관 기간이 7일 이하여야 통과 (삭제 지연까지 더하면 실제로는 최대 10일 남을 수 있음)
점검: /app/answers/debug: 만료 없음(저장이 계속 늘어요)
점검: /app/answers/debug: 원문 프롬프트가 허용 7일보다 오래 남아요(만료 없음)
```

## 이 실습이 확인하지 못하는 것

- 마스킹은 형식 규칙뿐이라 이름·주소는 잡지 못합니다(테스트가 이를 확인해요). 실제 번호인지 검증하지도 않아요. 서비스에 그대로 쓰지 마세요.
- 보관 기간 허용 값 목록은 CloudWatch Logs 공식 문서 기준(2026-11-14 확인)입니다. 바뀔 수 있어요.
- 개인정보 관련 법령의 보관 기준은 확인하지 않았습니다.
- 한국 형식 번호를 CloudWatch Logs 관리형 식별자가 지원하는지는 확인하지 못했어요.
- 전화·이메일·카드·주민번호 형식 값은 모두 가짜입니다.
