# 5장. 정적 호스팅과 Supabase: 서버 없이 로그인·DB·파일 해결하기 (`ch05_static_supabase`)

웹 챕터: `math-ai-matrix/tracks/webapp/05-static-supabase.html`

카드뉴스 편집기를 "서버를 직접 운영하지 않는" 구성으로 옮긴다. 화면 파일은 정적 호스팅(Vercel·Netlify)에 올리고, 로그인·카드 저장·그림 파일·AI 배경 만들기는 Supabase의 Auth, Postgres+RLS, Storage, Edge Functions가 맡는다.

1. **구성**: 브라우저 코드(`site/public/`)에는 프로젝트 주소와 공개용 키만 두고, AI 키는 Edge Function의 환경 변수에만 둔다. 실험 E0이 이 경계를 확인한다.
2. **정책**: `data/policies.sql`의 카드·사용량·스토리지 정책 일곱 개 중 스토리지 정책 네 개를 파이썬과 sqlite로 다시 만들어, 같은 요청 13개를 "정책 없음 / 정책 4개"로 보낸다(E1~E4). 카드 읽기·만들기 정책과 쓰기 정책 없는 사용량 표도 같은 방식으로 확인한다(E10).
3. **Edge Function**: `supabase/functions/generate-background/index.ts`와 같은 순서(토큰 확인 → 글 길이 → 하루 한도 → 비밀 키로 AI 호출 → 내 폴더에 저장 → 서명된 주소)를 파이썬 모형으로 돌린다(E5).
4. **무료 한도**: 공식 가격표(`data/pricing_2026-10-10.json`)와 가상 사용 패턴(`data/usage_assumptions.json`)으로 어느 한도가 언제 넘치는지, Pro 요금이 얼마인지 계산한다(E6). 벤더 종속(E7)과 일시 정지(E8)도 센다.

## 실행
```bash
uv run python webapp/ch05_static_supabase/experiments.py   # results/ch05.json 갱신
uv run pytest -q webapp/ch05_static_supabase
```
표준 라이브러리만 쓴다. 네트워크, Supabase 계정, 진짜 키 없이 돈다. `site/public/app.js`와 `index.ts`는 읽기용 예시이고 실행하지 않는다.

## 가짜 값 규칙
- 키·서명 비밀은 모두 `DEMO_`로 시작하는 가짜 값이다. 프로젝트 주소 `demo-project.supabase.example`, AI 주소 `ai-image.example`도 가상이다.
- 로그인 토큰은 HMAC-SHA256으로 서명한 JWT 모양의 모형이다. Supabase가 실제로 쓰는 서명 방식과 같다고 주장하지 않는다.
- `supabase/env.example`, `config.toml.example`은 실습 저장소에서 보이게 하려고 이름을 바꿨다. 실제로는 `supabase/functions/.env`(`.gitignore`에 넣음)와 `supabase/config.toml`이다.

## 파일
| 파일 | 역할 |
|---|---|
| `site/public/` | 정적 호스팅에 올리는 브라우저 코드(HTML·CSS·JS). 공개용 키만 있음 |
| `supabase/functions/generate-background/index.ts` | AI 배경 Edge Function 예시(`withSupabase`, `Deno.env.get`) |
| `supabase/config.toml.example`, `supabase/env.example` | `verify_jwt` 설정, 함수 비밀(가짜) |
| `data/policies.sql` | 카드 2개·사용량 1개·스토리지 4개 정책과 버킷 두 개 |
| `data/pricing_2026-10-10.json` | Supabase·Vercel·Netlify 한도·가격·넘쳤을 때 일과 출처 |
| `data/usage_assumptions.json` | 활성 사용자 1명의 한 달 사용량(가상), 가상 AI 단가 |
| `data/sources_2026-10-10.json` | 정책·구성 사실의 출처 URL, 원문 인용, 확인일 |
| `webapp_ch05_rules.py` | 스토리지·카드·사용량 정책 모형, 도우미 함수 세 개, sqlite 대조 |
| `webapp_ch05_edge.py` | Edge Function 모형(토큰, 하루 한도, 가짜 AI, 서명된 주소, 호출 수 세기) |
| `webapp_ch05_costs.py` | 무료 한도·넘치는 달·Pro 요금·Netlify 크레딧 계산(분수) |
| `experiments.py` | E0~E10 (E9 연습 답·중간 계산, E10 카드·사용량 표 정책) |
| `checks/ch05.csv` | 내 앱의 한도 기록 양식(예시 행은 모형 값) |

## 한계
- 정책 모형은 Postgres·Supabase가 아니다. 허용 정책의 OR, default deny, USING/WITH CHECK, 비밀 키 건너뛰기, 공개 버킷, upsert 조건만 옮겼다. 파일 이동·복사, 버킷 단위 크기·형식 제한, 이미지 변환은 다루지 않는다.
- Edge Function 모형은 플랫폼이 `verify_jwt`로 막은 호출을 요금에 세지 않았다. 문서는 "응답 코드와 상관없이 센다"고만 적어서, 플랫폼 단계에서 막힌 호출이 세는지는 확인하지 못했다([확인 필요]).
- `index.ts`의 `.eq()`, `.maybeSingle()`, `.upsert()` 같은 표 호출 모양은 이번에 문서로 확인하지 않았다([확인 필요]). 쓰기 전에 supabase-js 참조 문서를 본다.
- 비용 계산은 가상 사용 패턴 하나에 대한 것이다. 스토리지는 달 끝 크기로, egress는 모두 캐시 안 된 전송으로 셌다(둘 다 실제보다 넉넉하게). Vercel Pro의 전송량 초과 요금은 문서끼리 표기가 달라 계산하지 않았다. 원화 환산은 하지 않았다.
- 일시 정지 규칙(7일 연속 DB 요청 0이면 위험)은 모형 가정이다. 문서는 "a few user requests to the database each day"면 대개 충분하다고만 적는다.

확인일 2026-10-10.
