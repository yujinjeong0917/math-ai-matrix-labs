# 세 도구를 잇는 규칙 (6장, 관통 프로젝트 "동네 스터디룸 예약")

확인일 2026-10-08. 1장 등록부(정보 항목 / 원본 위치 / 참조하는 곳)를 도구 사이 흐름까지 넓힌 표다.
값은 `productivity_ch06_registry.py`의 `ITEMS`와 같고, `productivity_ch06_graph.validate()`가 아래 네 규칙을 검사한다.

## 원리 블록 (도구가 바뀌어도 그대로)
1. 정보 항목마다 원본은 하나. 그래프에서 들어오는 간선이 0인 위치가 원본 하나뿐이어야 한다.
2. 모든 위치는 값을 한 곳에서만 받는다. 두 곳에서 받으면 어느 쪽이 맞는지 정할 수 없다.
3. 고리 금지. A -> B와 B -> A를 같이 두는 양방향 동기화는 원본을 둘로 만든다(실험 E6).
4. 원본에서 모든 위치에 닿아야 한다. 끊긴 사본은 아무도 고치지 않는다.
5. 간선마다 갱신 방식을 적는다. 자동 참조(링크·임베드·수식·relation·컴포넌트 인스턴스)를 먼저 쓰고,
   안 되는 곳만 사람 확인으로 남긴다. 사람 확인 간선이 놓칠 수 있는 자리의 목록이다.
6. 고치려면 원본에서. 참조하는 곳에서 고치고 싶으면 원본 담당에게 댓글로 요청한다(권한 표가 이걸 강제한다).

## 규칙 표 (항목 12개, 간선 22개: 자동 참조 9, 사람 확인 11, 자동화 후보 2)
| 항목 | 정보 | 원본 | 받는 곳 | 갱신 방식 | 주기 | 책임자 | 변경 알림 |
|---|---|---|---|---|---|---|---|
| I01 | 취소 규정 문구 | `notion:Specs/S02.취소규정` | `figma:Components/취소안내` | 사람 확인 | 변경 당일 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I01 | 취소 규정 문구 | `figma:Components/취소안내` | `figma:S02_상세.취소안내` | 자동 참조 | 즉시 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I01 | 취소 규정 문구 | `figma:Components/취소안내` | `figma:S04_완료.취소안내` | 자동 참조 | 즉시 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I01 | 취소 규정 문구 | `notion:Specs/S02.취소규정` | `sheets:metrics!안내` | 자동 참조 | 즉시 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I02 | 취소율 지표 정의 | `notion:Specs/지표정의.취소율` | `sheets:metrics!B2수식` | 사람 확인 | 변경 당일 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I03 | 규정 변경 기준값 | `notion:Decisions/D04.기준` | `sheets:metrics!threshold` | 자동화 후보 | 1시간마다 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I04 | 지점 목록 | `sheets:branches` | `sheets:clean!lookup` | 자동 참조 | 즉시 | 하린 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I04 | 지점 목록 | `sheets:branches` | `notion:Projects.지점수` | 사람 확인 | 변경 당일 | 하린 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I05 | 운영 시간 | `notion:Specs/S02.운영시간` | `figma:S02_상세.운영시간` | 사람 확인 | 변경 당일 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I05 | 운영 시간 | `notion:Specs/S02.운영시간` | `sheets:capacity!hours` | 사람 확인 | 변경 당일 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I06 | 시간당 요금 | `sheets:lookup!price` | `figma:S03_예약.요금` | 사람 확인 | 변경 당일 | 하린 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I06 | 시간당 요금 | `sheets:lookup!price` | `notion:Specs/S03.요금` | 자동 참조 | 즉시 | 하린 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I07 | 예약 화면 프레임 | `figma:S03_예약` | `notion:Tasks.Figma링크` | 자동 참조 | 즉시 | 민지 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I08 | D10 확정(전화 인증 철회) | `notion:Decisions/D10.상태` | `notion:Tasks/T17-T18` | 자동화 후보 | 1시간마다 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I08 | D10 확정(전화 인증 철회) | `notion:Decisions/D10.상태` | `figma:S03_예약.전화칸` | 사람 확인 | 변경 당일 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I09 | 예약 상태 표준값 | `sheets:lookup!status` | `sheets:clean!status` | 자동 참조 | 즉시 | 하린 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I09 | 예약 상태 표준값 | `sheets:lookup!status` | `notion:Specs/지표정의.상태목록` | 사람 확인 | 변경 당일 | 하린 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I10 | 예약 버튼 문구 | `notion:Specs/S03.버튼문구` | `figma:Components/버튼` | 사람 확인 | 변경 당일 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I10 | 예약 버튼 문구 | `figma:Components/버튼` | `figma:S03_예약.버튼` | 자동 참조 | 즉시 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I11 | 주간 취소율 값 | `sheets:metrics!B2값` | `notion:Projects.주간리포트` | 사람 확인 | 변경 당일 | 하린 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I12 | 런칭일 | `notion:Decisions/D09.날짜` | `notion:Tasks/T19-T20` | 자동 참조 | 즉시 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |
| I12 | 런칭일 | `notion:Decisions/D09.날짜` | `sheets:metrics!기간` | 사람 확인 | 변경 당일 | 서연 | 변경 기록 + 받는 곳 담당에게 멘션 |

도구 사이 간선 가운데 자동 참조는 3개뿐이다(Sheets -> Notion 임베드, Notion -> Sheets 링크, Figma -> Notion 프레임 링크).
Notion 문구를 Figma 화면에 넣는 길(4개)은 모두 사람 확인이다.

## 권한 표 (단계 이름은 각 도움말 기준, 확인일 2026-10-08)
| 사람 | 역할 | Figma (Can view / Can edit) | Notion (Full access / Can edit / Can edit content / Can comment / Can view) | Google Sheets (Viewer / Commenter / Editor) |
|---|---|---|---|---|
| 민지 | 디자인, 화면 원본 | Can edit | Can comment | Viewer |
| 서연 | 기획, 결정·스펙 원본 | Can view | Full access | Commenter |
| 준호 | 개발 | Can view | Can edit | Viewer |
| 하린 | 운영, 숫자 원본 | Can view | Can comment | Editor |
| 가져오기 스크립트 | raw·clean 붙이기 | 없음 | 없음 | Editor |

- 원칙: 원본(`raw_*`·`branches`·`lookup` 시트, Notion `Decisions` 확정 항목, Figma `01 Components`)은 편집자를 최소로 둔다.
- Figma Can view도 댓글은 달 수 있다(도움말). 그래서 보기 권한만 있어도 원본 담당에게 요청을 남길 수 있다.
- 정리 시트(clean)는 사람이 고치지 않는다(5장). 시트 보호의 "Restrict who can edit this range"로 막는다.
- 권한으로 못 막는 것: 원본이 아닌 곳이라도 편집 권한이 정당하게 있는 사람의 수정(예: 디자이너가 Figma 컴포넌트에서 문구를 바꿈, 이벤트 E01).

## 버전·변경 기록 규칙
- 모든 변경은 Notion `Decisions` 또는 변경 기록 데이터베이스에 한 줄(날짜, 누가, 무엇을, 왜, 영향받는 위치). 페이지 기록이 아니라 데이터베이스 행이라 요금제 보존 기간과 상관없이 남는다고 본다(해석, 도움말에 직접 적힌 말은 아님).
- 지표 정의를 바꾸기 직전: Google Sheets에서 버전에 이름 붙이기(오른쪽 위 Last edit에서 버전을 고르고 More > Name this version). 스프레드시트 하나에 이름 붙인 버전은 15개까지라는 한도가 도움말에 있다.
- 개발자에게 넘기기 직전: Figma에서 Save to Version History로 이름과 설명을 단다. Starter 팀은 버전 기록을 30일만 볼 수 있다.
- Notion 페이지 기록은 Free 7일, Plus 30일, Business 90일(도움말). 넘겨준 뒤 2주가 지나면 Free 요금제의 페이지 기록으로는 "누가 언제"를 못 찾을 수 있다.
- Sheets의 셀 편집 기록에는 추가·삭제한 행, 서식 변경, 수식이 바꾼 값이 안 보일 수 있다(도움말: "Some changes might not show up"). 스크립트가 붙인 지점(E04)은 변경 기록 행으로 남긴다.

## 자동화 후보 3개 (설계만, 실제 연결은 독자 요금제·권한에서 되는 것만)
| | 무엇 | 수단 | 공식 문서에 적힌 함정 | 대비 |
|---|---|---|---|---|
| A1 | Sheets checks가 FALSE면 알림 | Apps Script 설치형 트리거 | 스크립트 실행과 API 요청은 트리거를 실행하지 않는다. 설치형 트리거는 만든 사람 계정으로 돌고, 실패하면 요약 메일이 간다(받는 사람은 문서에 따로 없고, 만든 사람으로 읽음: 해석) | onEdit 대신 시간 트리거. 지표 시트 맨 위에 "A1 마지막 실행 24시간 안" 칸(C13). 넘겨줄 때 만든 사람을 바꾸고 다시 만든다 |
| A2 | 결정이 확정되면 할 일 만들기 + 담당자 알림 | Notion 데이터베이스 자동화(유료 요금제) | 자동화가 만든 페이지는 다른 자동화를 실행하지 않는다 | 알림을 따로 두지 않고 같은 자동화 안의 동작으로 넣는다 |
| A3 | Notion 기준값을 Sheets 기준 칸으로 | API 스크립트(시간 트리거) | 이름으로 속성을 찾으면 이름이 바뀔 때 멈춘다. 웹훅은 순서가 바뀔 수 있고 내용이 없다. 잦은 변경은 묶어서 하나로 보낸다 | 속성 id로 찾는다(이름이 바뀌어도 그대로). 웹훅은 신호로만 쓰고 API로 최신 값을 다시 읽는다. C13에 "A3 마지막 성공" 칸 |

"Figma 프레임 링크가 없는 할 일"은 자동화가 아니라 Notion 보기(필터: Figma 링크가 비어 있음)로 충분하다. 깨질 부품이 없다.

## 넘겨준 기록 2건 (양식)
1. 디자인 -> 개발 (3장 체크리스트): 날짜, 넘긴 사람, 받은 사람, Figma 버전 이름, 체크리스트 칸 수 / 채운 칸 수, 규칙 표에서 Figma로 가는 사람 확인 간선 5개를 하나씩 대조한 결과.
2. 운영 -> 기획 (5장 checks): 날짜, checks 모두 TRUE인지, C13 자동화 마지막 실행, 이번 주 변경 기록 행 수 = 실제 변경 수인지.
`checks/ch06.csv`에 같은 칸을 두고, 모형으로 채운 예시는 `data/handoff_records.json`에 있다.

## 근거 (확인일 2026-10-08)
- Apps Script, Simple triggers / Installable triggers: https://developers.google.com/apps-script/guides/triggers , https://developers.google.com/apps-script/guides/triggers/installable
- Notion, Database automations: https://www.notion.com/help/database-automations
- Notion API, Webhooks 이벤트 전달: https://developers.notion.com/reference/webhooks-events-delivery , 속성 id: https://developers.notion.com/reference/property-object
- Figma REST API, Webhooks 이벤트: https://developers.figma.com/docs/rest-api/webhooks-events/
- 권한: https://help.figma.com/hc/en-us/articles/1500007609322 , https://www.notion.com/help/sharing-and-permissions , https://support.google.com/docs/answer/2494822
- 버전 기록: https://help.figma.com/hc/en-us/articles/360038006754 , https://support.google.com/docs/answer/190843 , https://www.notion.com/help/duplicate-delete-and-restore-content
