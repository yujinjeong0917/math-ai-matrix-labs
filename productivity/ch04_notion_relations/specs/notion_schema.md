# Notion 데이터베이스 4개 스키마 (4장)

확인일 2026-10-08. 메뉴 이름과 계산 종류는 Notion 도움말 "Relations & rollups"(https://www.notion.com/help/relations-and-rollups)과
"Formula syntax"(https://www.notion.com/help/formula-syntax) 기준이다. 템플릿 갤러리의 기존 템플릿을 옮기지 않고 새로 정했다.

## Projects
| 속성 | 종류 | 메모 |
|---|---|---|
| 이름 | title | |
| 상태 | status | 준비 / 진행 / 런칭 / 종료 |
| 기간 | date | |
| 결정 | relation → Decisions | 양방향(Decisions 쪽 "관련 프로젝트") |
| 결정 수 | rollup | 결정 relation, Count all |
| 할 일 | relation → Tasks | |
| 미완료 할 일 수 | formula | `prop("할 일").filter(current.prop("상태") != "완료").length()` |

## Decisions (결정 로그)
| 속성 | 종류 | 메모 |
|---|---|---|
| 제목 | title | "D04 취소 규정 변경"처럼 번호를 붙인다 |
| 날짜 | date | |
| 결정 내용 / 고려한 대안 / 이유 | text | 이유가 빠지면 이 데이터베이스를 만드는 의미가 없다 |
| 결정자 | person | 빈칸 검사 대상 |
| 상태 | select | 제안 / 확정 / 번복 |
| 관련 프로젝트 | relation → Projects | |
| 영향받는 할 일 | relation → Tasks | 양방향(Tasks 쪽 "근거 결정") |
| 영향받는 스펙 | relation → Specs | 양방향. 할 일을 거쳐 추론하지 않고 직접 잇는다(연결 함정, 4장 7절) |
| 번복한 결정 / 번복된 결정 | relation → Decisions(자기 참조) | Two-way relation을 켜서 두 속성으로 나눈다 |
| 미완료 할 일 수 | formula | `prop("영향받는 할 일").filter(current.prop("상태") != "완료").length()` |

rollup 계산 종류(Count all, Count values, Count unique values, Count empty, Count not empty, Percent empty, Percent not empty, Sum 등)에는
도움말 기준으로 "조건에 맞는 것만 세기"가 없다. 그래서 미완료 수는 수식(formula)으로 센다.

## Tasks
| 속성 | 종류 | 메모 |
|---|---|---|
| 제목 | title | |
| 상태 | status | 할 일 / 진행 / 검토 / 완료 |
| 담당 | person | |
| 마감 | date | |
| 근거 결정 | relation → Decisions | 빈칸 검사 대상 |
| 관련 스펙 | relation → Specs | |
| Figma 프레임 링크 | url | 2·3장의 프레임 |
| 마지막 수정 | last edited time | 행마다 따로 남는다(일반 체크리스트와 다른 점) |

## Specs
| 속성 | 종류 | 메모 |
|---|---|---|
| 제목 | title | |
| 화면 번호 | select | S01~S04 |
| 문구 원본 여부 | checkbox | 3장 체크리스트의 "문구 원본은 Notion 스펙 페이지" |
| 마지막 수정 | last edited time | |
| 관련 결정 | relation → Decisions | |
| 열린 할 일 수 | formula | Tasks relation에 같은 filter 수식 |

## 보기(view)
- 결정 로그 타임라인(날짜 기준)
- 할 일 상태별 보드
- "스펙 수정 뒤 갱신 안 된 할 일": Tasks에서 상태 = 완료, 마지막 수정 < 관련 스펙의 마지막 수정
  (Notion 필터 하나로 두 데이터베이스의 시각을 바로 비교할 수 있는지는 확인하지 못했다 [확인 필요]. 수식 속성으로 비교 결과를 만들어 그 속성으로 거른다.)
- 빈칸 검사: 근거 결정이 빈 할 일, 결정자가 빈 결정, 영향받는 스펙이 빈 결정

## 검증 절차
1. 질문 5개를 필터·rollup·수식만으로 답한다.
2. 미완료 할 일 수를 손으로 필터한 개수와 대조해 모두 일치하는지 본다.
3. 번복된 결정이 원래 결정과 양방향으로 연결돼 있는지 본다(D07 페이지에서 D10이 보여야 한다).
4. 근거 결정이 빈 할 일이 0개인지 본다.
