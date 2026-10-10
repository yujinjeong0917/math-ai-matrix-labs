# 3장. 바이브코딩 앱이 자주 놓치는 보안 다섯 가지 (`ch03_vibe_security`)

웹 챕터: `math-ai-matrix/tracks/webapp/03-vibe-security.html`

1장 카드뉴스 편집기에 로그인, 카드 저장(DB), 관리 화면, 배경 그림 업로드, AI 배경 만들기를 붙인 작은 가상 앱(sqlite + 표준 라이브러리 `http.server`)을 만들고, AI로 빠르게 만든 앱이 자주 놓치는 다섯 가지를 각각 "고치기 전 / 고친 뒤"로 같은 입력에 돌려 비교한다. 공격 방법이 아니라 점검 목록과 방어 코드가 목적이다.

1. **RLS(행 수준 보안)**: `data/policies.sql`의 Supabase 정책과 같은 규칙을 sqlite + 파이썬으로 다시 만들고, 끔 / 켜기만 함 / 정책까지 세 상태에서 로그인 안 한 사람·alice·bob이 읽고 쓰는 결과를 센다.
2. **AI 비밀 키를 브라우저 번들에 넣기**: 고치기 전 번들에는 가짜 비밀 키가 있고, 고친 뒤에는 서버가 대신 부른다(자세한 점검은 2장).
3. **주소만 알면 열리는 관리 화면**: 링크만 숨긴 것과 서버가 세션의 역할을 확인하는 것(401 / 403 / 200).
4. **업로드 검증**: 크기, 이름 규칙, 확장자 허용 목록, Content-Type, 매직 바이트. 통과한 파일은 서버가 만든 이름으로 공개 폴더 밖에 둔다.
5. **사용량 제한**: 사용자별 토큰 버킷(용량 5, 12초에 1개), 429와 Retry-After.

## 실행
```bash
uv run python webapp/ch03_vibe_security/experiments.py   # results/ch03.json 갱신
uv run pytest -q webapp/ch03_vibe_security
uv run python webapp/ch03_vibe_security/webapp_ch03_app.py before   # 고치기 전 앱 직접 띄우기(127.0.0.1:8003)
uv run python webapp/ch03_vibe_security/webapp_ch03_app.py after    # 고친 뒤 앱
```
표준 라이브러리만 쓴다. 서버는 `127.0.0.1`에만 묶고, 실험에서는 빈 포트에 잠깐 띄웠다 끈다. 사용량 제한은 가짜 시계(`FakeClock`)를 넣어 sleep 없이 매번 같은 결과를 낸다. 토큰 수와 시간은 분수(`fractions.Fraction`)로 센다.

직접 띄운 앱에서 확인할 때는 세션 토큰 `demo-session-alice`, `demo-session-bob`, `demo-session-admin`(가짜)을 `Authorization: Bearer ...` 머리글로 보낸다. 이 실습 앱과 내가 만든 앱에만 써 본다. 기록 양식은 `checks/ch03.csv`.

## 파일
| 파일 | 역할 |
|---|---|
| `webapp_ch03_app.py` | 다섯 가지를 켜고 끄는 가상 앱 서버(`BEFORE`, `AFTER`), 127.0.0.1 전용 요청 함수 |
| `webapp_ch03_rls.py` | cards 표, 정책 다섯 개의 파이썬판, 같은 select 정책의 sqlite WHERE 절 |
| `webapp_ch03_upload.py` | 업로드 검증, 매직 바이트 판정, 시험 파일 11개(직접 만든 1×1 PNG 포함) |
| `webapp_ch03_ratelimit.py` | 토큰 버킷, 사용자별 제한, 1장식 전체 횟수 세기(비교용), 시뮬레이션 |
| `app/` | 고치기 전/후 브라우저 번들(`public_before/`, `public_after/`)과 관리 화면 |
| `data/policies.sql` | Supabase(Postgres)용 RLS 정책 SQL |
| `data/sources_2026-10-10.json` | 정책·점검 항목·사례 숫자의 출처 URL, 원문 확인한 인용, 확인일 |
| `experiments.py` | E0~E6 |
| `checks/ch03.csv` | 내 앱 점검 기록 양식(예시 행은 모형 값) |

## 한계
- RLS 모형은 Postgres가 아니다. 허용(permissive) 정책의 OR, default deny, USING / WITH CHECK, 보이지 않는 행은 조용히 0행이라는 규칙만 옮겼다. 제한(restrictive) 정책, 열 단위 권한(grant), 함수·뷰를 거치는 접근, `auth.uid()`를 꺼내는 토큰 검증은 흉내 내지 않았다. 거부 응답을 403으로 둔 것도 모형에서 정한 값이다.
- 업로드 검사는 앞부분 바이트만 본다. 그림을 다시 그려 저장(재인코딩)하거나 바이러스 검사를 하는 단계는 표준 라이브러리로 할 수 없어 넣지 않았다. JPEG·WebP 시험 파일은 머리만 있는 자리표시다.
- 사용량 제한 상태는 서버 메모리에 있다. 서버가 여러 대이거나 다시 켜지면 버킷이 나뉘거나 비워진다. 비용(1장 50원)은 가상 단가다.
- 키·토큰·이메일·카드 내용은 모두 실습용으로 만든 가짜 값이다.

확인일 2026-10-10.
