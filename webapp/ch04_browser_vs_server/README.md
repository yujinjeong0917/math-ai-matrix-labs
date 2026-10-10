# 4장. 브라우저 기반과 서버 기반: 기능마다 어디서 돌릴까 (`ch04_browser_vs_server`)

웹 챕터: `math-ai-matrix/tracks/webapp/04-browser-vs-server.html`

1장 카드뉴스 편집기의 기능 7개(편집, 미리보기, 템플릿 고르기, AI 배경 생성, 최종 PNG 출력, 임시 저장, 보관 저장)를 "보이면 안 되는 것이 있나 → 기기를 바꿔도 남아야 하나 → 얼마나 빨리, 얼마나 자주"의 순서로 판단해 브라우저와 서버로 나눈다. 그다음 미리보기나 최종 PNG 렌더를 서버로 옮겼을 때 요청 수, 대기 시간, 한 달 서버비(Vercel Pro 서울 리전, Netlify 크레딧)가 어떻게 바뀌는지 센다. 브라우저 저장소(localStorage, IndexedDB)에 무엇이 얼마나 들어가고 언제 지워지는지도 MDN 규칙으로 모형을 만든다.

1. 판단표: `webapp_ch04_decide.py` (규칙 1 비밀 → 서버, 규칙 2 기기를 바꿔도 남아야 함 → 서버, 규칙 3 나머지 → 브라우저)
2. 요청·대기·비용: `webapp_ch04_cost.py` (설계 네 가지: PNG까지 브라우저 / 판단표대로 / 미리보기도 서버 / 미리보기 서버 + 300ms 디바운스)
3. 저장소: `webapp_ch04_storage.py` (localStorage 5 MiB에 들어가는 초안·그림 수, base64, IndexedDB 한도, 지워지는 조건)

## 실행
```bash
uv run python webapp/ch04_browser_vs_server/experiments.py   # results/ch04.json 갱신
uv run pytest -q webapp/ch04_browser_vs_server
uv run python webapp/ch04_browser_vs_server/webapp_ch04_decide.py   # 판단표만 보기
```
표준 라이브러리만 쓴다. 네트워크·계정·API 키 없이 돌고, 돈은 분수(`fractions.Fraction`)로 세서 결과가 매번 같다.

## 파일
| 파일 | 역할 |
|---|---|
| `data/features.json` | 기능 7개의 속성(건드리는 비밀, 기기를 바꿔도 남아야 하나, 카드 한 장당 쓰는 횟수, 기다려도 되는 시간). 가정 |
| `data/assumptions.json` | 사용자 수, 왕복 지연, 회선 속도, 서버 계산 시간, 응답 크기, AI 가상 단가(1장 50원, 3장과 같음), 환율. 모두 가정 |
| `data/prices_2026-10-10.json` | Vercel·Netlify 공식 가격 문서에서 옮긴 단가와 URL, 확인일 |
| `data/sources_2026-10-10.json` | 저장소 한도·삭제 조건, 반응 시간 기준, 브라우저 검사 우회 등의 출처와 인용 |
| `data/draft_example.json` | 임시 저장할 카드 초안 예시 |
| `webapp_ch04_decide.py` | 판단표 |
| `webapp_ch04_cost.py` | 타자·디바운스 모형, 대기 시간, Vercel·Netlify 비용 |
| `webapp_ch04_storage.py` | 브라우저 저장소 모형 |
| `experiments.py` | E0~E11 (E11은 본문에 나오는 중간 계산값) |
| `checks/ch04.csv` | 내 앱에서 직접 재 볼 기록 양식(예시 행은 모형 값) |

## 한계
- 모든 시간·크기·사용량은 가정이다. 실제 대기 시간은 사용자 회선, 서버 리전, 처음 켜질 때의 지연(콜드 스타트, 이 모형에 없음)에 따라 달라진다.
- Vercel 메모리 요금은 요청 하나가 인스턴스 하나를 혼자 쓴다고 둔 위쪽 어림이다. 실제로는 여러 요청이 한 인스턴스를 나눠 쓴다. 함수 요청도 CDN 요청으로 세고, 방문자 쪽 전송은 Pro에 들어 있는 Flat Rate CDN 등급으로 덮었다. 호출 단가 $0.60/100만은 문서 본문 예시 문장에서 옮겼고, 같은 문서의 표에는 'N/A'로 적혀 있다.
- Netlify Pro는 3,000~20,000크레딧 단계를 고르게 되어 있는데, 가장 낮은 단계를 월 $20로 보고 넘치는 크레딧은 자동 충전 값으로 셌다. 높은 단계 가격은 확인하지 못했다.
- Flat Rate CDN은 새 Pro 팀에서 기본으로 켜진다(변경 기록). 예전부터 쓰던 팀이 켜지 않았다면 CDN 요청과 방문자 쪽 전송이 단가대로 따로 붙어 이 모형보다 비싸다.
- localStorage를 글자당 2바이트로 센 것은 가정이다. 브라우저가 한도를 정확히 어떻게 세는지는 이번에 읽은 문서에서 확인하지 못했다.
- 1장의 AI 배경은 491바이트짜리 흉내 SVG였지만, 이 장의 비용 계산은 실제 크기에 가깝게 1.5MB PNG로 둔다.

확인일 2026-10-10.
