# 9장. 누구나 만드는 시대에 남는 것 (`ch09_ai_coding_era`)

웹 챕터: `math-ai-matrix/tracks/webapp/09-ai-coding-era.html`

트랙 마지막 장이다. 1~8장 실습 결과를 모아 "출시 전 점검표" 하나로 만들고, 관통 프로젝트인 카드뉴스 편집기의 두 판(AI로 처음 빨리 만든 첫 판, 고친 출시 후보)에 같은 점검표를 돌린다.

1. 점검 항목 31개를 장별로 모은다(1장 보이는 것, 2장 키, 3장 다섯 가지, 4장 어디서 돌릴지, 5장 RLS·한도, 6·7장 포장·서명, 8장 플랫폼·수수료, 9장 운영).
2. 항목마다 확인 방법을 정한다. **다시 검사**(앞 장 파일을 앞 장 모듈의 순수 함수로 다시 판정), **결과 인용**(앞 장 `results/chNN.json` 값을 위치와 함께 읽음), **사람 확인**(파일로 답할 수 없어 참고값만 붙임).
3. 두 판의 통과 수를 세고, 인용·다시 검사한 값이 앞 장 결과와 같은지 대조한다.

## 실행
```bash
uv run python webapp/ch09_ai_coding_era/experiments.py   # results/ch09.json, results/checklist_ch09.csv 갱신
uv run pytest -q webapp/ch09_ai_coding_era
uv run python webapp/ch09_ai_coding_era/webapp_ch09_checklist.py   # 점검표만 보기
```
표준 라이브러리만 쓴다. 서버를 띄우지 않고, 바깥 네트워크와 API 키 없이 돈다. 같은 저장소의 `webapp/ch01`~`ch07` 폴더가 있어야 한다(그 폴더의 파일과 결과를 읽는다).

## 두 판
| 판 | 모은 것 |
|---|---|
| 첫 판 | 1장 `editor/public`(지도 파일 포함), 2장 `project/dist_mistake`, 3장 `app/public_before`와 before 결과, 6장 `data/manifests/broken_first_draft.json`, 7장 `project/cardnews_mistake` |
| 출시 후보 | 2장 `project/dist_good`, 3장 after 결과, 4장 판단표, 5장 정책·함수·`site/public`, 6장 `pwa/public`, 7장 `project/cardnews_good` |

## 파일
| 파일 | 역할 |
|---|---|
| `webapp_ch09_checklist.py` | 점검표 만들기. 2장 `webapp_ch02_scan`, 6장 `webapp_ch06_manifest`, 7장 `webapp_ch07_config`·`webapp_ch07_signing`을 불러 쓴다 |
| `experiments.py` | E0~E6 |
| `data/ch08_manual_2026-10-10.json` | 8장 사람 확인 항목 4개와 공식 문서 숫자(수수료, 계정 비용, 비공개 테스트 요건). 8장 `results/ch08.json`이 있으면 그 계산 값을 참고값으로 덧붙인다 |
| `data/market_2026-10-10.json` | 본문의 시장 숫자(회사·조사 기관이 밝힌 수치, 출처 URL) |
| `data/sources_2026-10-10.json` | 이 장에서 새로 확인한 사실의 출처와 확인일 |
| `results/ch09.json` | 결과. `meta.read_results`에 읽은 앞 장 JSON의 SHA-256을 적는다 |
| `results/checklist_ch09.csv` | 점검표를 표로 |
| `checks/ch09.csv` | 사람 확인 항목 11개를 독자가 자기 앱으로 적는 양식 |

## 한계
- 8장 항목(플랫폼, 수수료, 계정 비용, 비공개 테스트)은 고르는 판단이라 모두 사람 확인으로 두고, 8장 결과의 계산 값(4,900원 한 건에 남는 돈, 첫해 고정비, 선택 도구 결과)은 참고값으로만 인용한다. 8장 결과 모양이 바뀌면 참고값은 `null`이 된다.
- 두 판은 실제 한 저장소가 아니라 각 장의 예시 폴더를 모은 것이다. 3장·4장 일부 항목은 다시 검사하지 않고 결과를 인용했다.
- 2장 모양 점검기는 3장 첫 판의 가짜 AI 키(`DEMO_SECRET_NOT_REAL_0003`)를 찾지 못한다. 실습용 표식이 2장 규칙과 다르기 때문이다. 3장 항목(이름으로 센 값)이 대신 잡는다. 점검기 하나로 끝내지 않는 이유다.
- 앞 장 결과가 바뀌면 `test_saved_results_match_live`가 실패한다. `experiments.py`를 다시 돌려 숫자를 갱신한다.
- 모든 키·비밀번호·주소는 실습용 가짜 값이다.

확인일 2026-10-10.
