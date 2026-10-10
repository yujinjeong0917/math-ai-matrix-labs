# 2장. API 키 노출: 공개용 키와 비밀 키 구분하기 (`ch02_api_keys`)

웹 챕터: `math-ai-matrix/tracks/webapp/02-api-keys.html`

1장에서 만든 카드뉴스 편집기에 "내 카드 저장하기"(Supabase)를 붙이면서, 브라우저 코드에 들어가도 되는 키와 서버에만 둬야 하는 키를 나눈다.

1. 회사 공식 문서로 확인한 키 구분표(`data/key_rules_2026-10-10.json`)를 만든다.
2. 배포 폴더에서 키 모양을 찾는 점검기(`webapp_ch02_scan.py`)로 1장 편집기와 이번 장의 빌드 두 판을 훑는다.
3. 공개용 키만 새어 나갔을 때 RLS(행 수준 보안)가 없음/있음에 따라 카드 몇 장이 읽히는지 sqlite 모형으로 센다.

## 실행
```bash
uv run python webapp/ch02_api_keys/experiments.py                 # results/ch02.json 갱신
uv run pytest -q webapp/ch02_api_keys
uv run python webapp/ch02_api_keys/webapp_ch02_build.py           # env.*나 src/를 고쳤다면 dist 다시 만들기
uv run python webapp/ch02_api_keys/webapp_ch02_scan.py <폴더> real # 내 배포 폴더 점검(비밀이 있으면 종료 코드 1)
```
표준 라이브러리만 쓴다. 네트워크와 진짜 키 없이 돈다.

## 가짜 값 규칙
- 실습 폴더의 키는 모두 `DEMO_`로 시작하는 가짜 표식(`DEMO_SUPABASE_PUBLISHABLE_0001` 등)이다. 어떤 회사의 키 모양도 흉내 내지 않았다.
- 점검기는 `demo` 모드에서 이 표식을, `real` 모드에서 공식 문서에 적힌 접두사(`sb_secret_`, `pk_live_` 등)를 찾는다.
- 진짜 접두사 규칙은 테스트가 실행 중에 만든 문자열로만 시험하고 파일로 남기지 않는다. `test_no_real_key_shapes_on_disk`가 이 폴더 전체에 진짜 모양의 키가 없는지 확인한다.
- 설정 파일 이름을 `.env` 대신 `env.good`, `env.mistake`로 둔 것은 실습 저장소에서 보이게 하려는 것이다. 실제 앱에서는 `.env`를 `.gitignore`에 넣는다.

## 파일
| 파일 | 역할 |
|---|---|
| `data/key_rules_2026-10-10.json` | 키 구분표: 회사, 키 이름, 공식 접두사, 공개용/비밀, 둘 곳, 방어선, 출처 |
| `data/sources_2026-10-10.json` | 사실마다 출처 URL·확인일 |
| `project/env.good`, `project/env.mistake` | 빌드에 쓰는 환경 변수(가짜). mistake는 비밀 키 이름 앞에 `VITE_`를 붙인 판 |
| `project/src/app.js`, `project/src/app_mistake.js` | 편집기 저장 기능 원본 코드 두 판 |
| `project/dist_good/`, `project/dist_mistake/` | 빌드 결과(배포 폴더) |
| `project/supabase/policies.sql` | 카드 테이블 읽기 정책(Postgres·Supabase 문법 예시) |
| `webapp_ch02_build.py` | `VITE_` 이름만 값으로 바꿔 넣는 빌드 흉내 |
| `webapp_ch02_scan.py` | 접두사·옛 JWT의 role·이름으로 키를 찾는 점검기. 값은 접두사 뒤 6글자와 SHA-256 앞 16글자만 남긴다 |
| `webapp_ch02_rls.py` | RLS 모형(sqlite + 파이썬 대조) |
| `experiments.py` | E0~E5 |
| `checks/ch02.csv` | 내 앱의 키를 적어 보는 양식(예시 행은 모형 값) |

## 한계
- 점검기는 아는 모양과 이름만 찾는다. 0건은 "아는 모양이 없다"는 뜻이지 "비밀이 없다"는 뜻이 아니다. OpenAI·Anthropic·Google 키의 접두사는 오늘 읽은 공식 문서에서 확인하지 못해 모양 규칙에 넣지 않았다([확인 필요]).
- 빌드 흉내는 `import.meta.env.이름` 바꾸기만 한다. 진짜 Vite의 압축·묶기는 하지 않는다.
- RLS 모형은 Postgres가 아니다. 권한(grant) → service_role 건너뛰기 → RLS 꺼짐 → 정책 OR 순서만 따라 하고, 읽기(select)만 다룬다. 쓰기와 권한 회수는 3장에서 다룬다.

확인일 2026-10-10.
