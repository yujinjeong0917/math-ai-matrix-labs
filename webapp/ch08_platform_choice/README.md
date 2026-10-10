# 8장. 첫 출시 플랫폼 고르기: iOS, 안드로이드, 웹 (`ch08_platform_choice`)

웹 챕터: `math-ai-matrix/tracks/webapp/08-platform-choice.html`

7장에서는 행사장 태블릿을 우리가 골라서 APK로 충분했다. 이번에는 손님 휴대폰처럼 기기를 고를 수 없을 때, 관통 프로젝트(카드뉴스 편집기)의 첫 출시를 웹·안드로이드·iOS 중 어디로 할지 따진다.

1. 타깃 조건(나라, 유료/무료, 결제 종류, 필요한 기기 기능, 가진 장비)을 넣으면 세 플랫폼마다 조건별 `충족 / 미충족 / 확인 필요`와 근거 id를 내는 선택 도구. **점수·순위는 없다.** 언제나 웹, 안드로이드, iOS 순서로 돌려준다.
2. 월 매출 가정(구독 4,900원 × 200명)에서 스토어 수수료(애플 표준·소규모 사업자 프로그램·구독 1년 넘은 몫, 구글 플레이 15%, 한국 대체 결제·외부 결제)와 웹 결제(PG 카드 수수료)를 같은 기준으로 비교하는 계산기.
3. 첫해 고정비(개발자 계정, 맥이 없을 때 맥 값)와 그것을 메울 결제 건수, 100만 달러 기준선까지의 거리.

## 실행
```bash
uv run python webapp/ch08_platform_choice/experiments.py        # results/ch08.json 갱신
uv run pytest -q webapp/ch08_platform_choice
uv run python webapp/ch08_platform_choice/webapp_ch08_select.py  # 예시 타깃 하나의 조건 목록
```
표준 라이브러리만 쓴다. 네트워크 없이 돌고 결과는 매번 같다.

## 내 조건으로 돌려 보기
```python
import webapp_ch08_select as S
print(S.report(S.evaluate(S.Target(countries=("KR",), pricing="subscription", payment="digital",
                                   features=("push",), has_mac=False, has_android_phone=True))))
```
수수료는 `webapp_ch08_fees.channel_table(월 매출, load_fees(), share_later=0.4)`처럼 부른다. `share_later`는 구독 매출 중 유료 1년이 넘은 구독자에게서 나오는 몫(가정)이다.

## 계산의 기준(가정)
- 고객이 낸 금액은 부가세 10% 포함. 부가세(금액 ÷ 11)를 먼저 뗀다.
- 스토어 수수료는 부가세를 뗀 금액에 매긴다(애플 Schedule 2). 예외로 애플 한국 외부 결제 26%는 부가세 포함 금액에 매긴다(애플 문서의 "gross of any value-added taxes").
- PG 수수료는 고객이 낸 금액 전체에 매기고, 수수료에 붙는 부가세는 넣지 않는다.
- 대체 결제·외부 결제는 줄어든 스토어 수수료와 PG 수수료를 둘 다 낸다.
- 환율 1달러 = 1,400원, 맥 4년 사용은 이 장의 가정이다.
- 율을 확인하지 못한 경로(구글 한국 새 구조 등)는 0이 아니라 `None`으로 남긴다.

## 파일
| 파일 | 역할 |
|---|---|
| `webapp_ch08_select.py` | 선택 도구(조건 충족 목록, 고정 순서) |
| `webapp_ch08_fees.py` | 수수료·고정비 계산기 |
| `experiments.py` | E0~E6 |
| `data/fees_2026-10-10.json` | 수수료율·계정 비용·맥 가격·가정 값. 바뀌면 날짜가 다른 새 파일 |
| `data/sources_2026-10-10.json` | 사실 37개의 출처 URL·확인일, 확인하지 못한 것 7개 |
| `data/statcounter_2026-10-10/` | StatCounter 모바일 OS 월별 비율(2025-10~2026-09, 한국·미국·일본·전 세계). StatCounter 자료는 CC BY-SA 3.0 |
| `checks/ch08.csv` | 출시 직전에 다시 확인할 값 기록 양식(예시 행은 이 장의 값) |

## 한계
- 수수료 표는 한국 사용자에게 파는 경우다. 구글 플레이는 2026-12-31부터 한국에도 새 구조가 적용되고 결제 수수료는 아직 발표되지 않았다. 미국·EEA·영국·호주·일본은 이미 다른 구조다.
- 세금 처리는 단순화했다. 실제 정산은 세무 전문가와 각 스토어의 정산 보고서로 확인한다.
- StatCounter는 추적 코드가 붙은 사이트의 페이지뷰 비율이라 설치 기기 수나 결제 사용자 비율이 아니다. 2026-09처럼 한 달에 크게 움직이는 값도 있어 12개월 범위를 함께 본다.
- 선택 도구의 규칙은 2026-10-10에 읽은 문서를 옮긴 것이다. 기기 기능은 푸시·블루투스·카메라만 다룬다.

확인일 2026-10-10.
