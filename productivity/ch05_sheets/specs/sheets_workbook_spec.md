# Google Sheets 통합 문서 명세 (5장)

확인일 2026-10-08. 메뉴 이름은 Google Docs 편집기 도움말 기준이다.
데이터 검증(https://support.google.com/docs/answer/186103), 피벗 테이블(https://support.google.com/docs/answer/1272900),
시트·범위 보호(https://support.google.com/docs/answer/1218656), 이름 있는 범위(https://support.google.com/docs/answer/63175),
파일 한도(https://support.google.com/drive/answer/37603).

## 시트 구성
| 시트 | 하는 일 | 규칙 |
|---|---|---|
| `raw_bookings` | 예약 CSV를 붙여 넣기만 한다 | 붙여 넣기 전에 모든 열의 서식을 일반 텍스트(plain text)로 둔다(메뉴 경로는 도움말에서 확인하지 못함 [확인 필요]; 일반 텍스트로 두면 변환을 피한다는 근거는 Abeysooriya 등 2021). 붙인 뒤 Data > Protect sheets and ranges > "Restrict who can edit this range". 첫 행 머리글 고정 |
| `raw_branches` | 지점 목록 원본 | 같은 규칙 |
| `import_log` | 붙여 넣을 때마다 한 줄: 날짜, 보낸 쪽이 알려 준 행 수, 붙여 넣은 행 수 | 보낸 쪽 행 수는 파일 이름이나 메일에서 옮긴다 |
| `import_snapshot` | 붙여 넣은 직후의 지점×상태 건수 표를 값으로 복사해 둔 것(C8용, 선택) | 손으로 고치지 않는다 |
| `lookup` | 상태 입력값 -> 표준값(예약·취소·노쇼·완료), 지점 코드 목록 | 검증 규칙의 기준 목록 |
| `clean` | `raw_bookings`를 수식으로만 참조해 정리 | 손 입력 금지. 날짜 변환, 상태 표준화(lookup), 지점 확인, 이용 시간 0 초과 12 이하. 사유가 빈 행만 남긴다 |
| `excluded` | `raw_bookings`에서 뺀 행과 사유(날짜 모름·상태 모름·지점 모름·이용 시간 범위 밖) | 수식(FILTER)으로만 만든다. 원본 행 하나는 `clean`과 `excluded` 가운데 정확히 한 곳에 간다 |
| `pivot_branch_status` | `clean` 기준 피벗: 행 지점, 열 상태, 값 건수(COUNTA) | Insert > Pivot table |
| `metrics` | 지점별 취소율, 지점 평균 취소율, 전체 취소율, 노쇼율 | 범위는 `clean!A2:A`처럼 열 끝까지 연다. 지표 정의는 Notion Specs의 지표 정의 페이지가 기준(4장) |
| `checks` | 점검 수식. 모두 TRUE여야 결론을 낸다 | 아래 C1~C9 |

## 데이터 검증 (사람이 직접 입력하는 시트에만)
- 상태: Dropdown from a range = `lookup`의 표준값 네 개. "If the data is invalid"는 기본값(거부)을 둔다. "Show a warning"으로 바꾸지 않는다.
- 날짜: 유효한 날짜만. 지점: `lookup` 지점 목록. 이용 시간: 0보다 크고 12 이하의 숫자.
- 붙여 넣은 값에도 거부가 적용되는지는 도움말에서 확인하지 못했다 [확인 필요]. 그래서 붙여 넣는 시트(raw)는 검증 대신 clean과 checks로 본다.

## checks
| 이름 | 수식 모양 | 잡는 것 |
|---|---|---|
| C1 원본 행 = 정리 + 뺀 행 | `=COUNTA(raw!A2:A) = COUNTA(clean!A2:A) + COUNTA(excluded!A2:A)` | 정리 단계에서 행이 사라지거나 늘어남 |
| C2 피벗 합계 = 정리 행 | 피벗 총합계 = `COUNTA(clean!A2:A)` | 피벗 범위가 clean 전체를 덮지 않음 |
| C3 표준 밖 상태 0 | 뺀 사유 "상태 모름" 개수 = 0 | lookup에 없는 표기 |
| C4 날짜 변환 실패 0 | 뺀 사유 "날짜 모름" 개수 = 0 | `5일`처럼 달이 없는 날짜 |
| C5 분모 0인 지점 0 | 예약 0건인 지점 = 0 | 지점은 목록에 있는데 데이터가 없음 |
| C6 보낸 행 수 = 원본 행 | `import_log` 마지막 줄의 보낸 쪽 행 수 = 지금의 `COUNTA(raw_bookings!A2:A)` | 중간에서 잘린 파일, 보호를 풀고 raw에서 지운 행 |
| C7 목록 밖 지점 0 | 뺀 사유 "지점 모름" 개수 = 0 | 손으로 적은 합계 줄, 오타 지점 |
| C8 원본 지문 그대로 | (선택) `import_snapshot`의 지점×상태 건수가 지금 raw로 다시 센 표와 칸마다 같다 | 보호를 풀고 정렬·수정한 경우. 행 수·합계만 비교하면 열 하나 정렬은 못 잡는다(실험 E4: 합계 그대로). 파이썬 모형은 행 전체의 해시를 써서 더 넓게 잡고, 이 표는 지점·상태가 바뀐 경우만 잡는다 |
| C9 이용 시간 범위 밖 0 | 뺀 사유 "이용 시간 범위 밖" 개수 = 0 | 0 이하이거나 12를 넘는 이용 시간. `clean`이 행을 빼는 사유마다 점검이 하나씩 있어야 행이 조용히 빠지지 않는다 |

## 이 명세가 못 막는 것
- 목록 안의 틀린 값: 노쇼를 완료로 고르면 검증·점검이 모두 통과한다(실험 E7).
- C4가 FALSE일 때 그 행을 빼고 계산하면 지표가 기울 수 있다(실험 E5). FALSE면 결론을 내지 말고 원본을 고친다.
- 고유 고객 수(COUNTUNIQUE)는 지점별 값을 더하면 안 된다. 보존식 C2는 건수에만 쓴다(실험 E8).
- 이름 있는 범위도 정해진 칸을 가리킨다(도움말 예: "budget_total" = A1:B2). 그래서 이름을 붙여도 아래에 붙인 행은 들어가지 않을 것으로 본다 [해석](도움말에 직접 적힌 문장은 아님).
