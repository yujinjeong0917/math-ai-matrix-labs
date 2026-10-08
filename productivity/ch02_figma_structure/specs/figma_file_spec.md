# Figma 파일 구조 명세 (관통 프로젝트 "동네 스터디룸 예약")

확인일 2026-10-08. 메뉴 위치는 바뀔 수 있어 기능 이름으로 적었다. 근거 도움말은 맨 아래에 있다.

## 페이지
| 페이지 | 담는 것 |
|---|---|
| `Cover` | 파일 목적, 담당자, 마지막 확인일 |
| `00 Foundations` | 색·글자·간격 값(간격은 4의 배수만) |
| `01 Components` | 버튼, 입력창, 목록 셀, 상단바의 main component |
| `02 Screens` | 화면 프레임 S01~S04 |
| `03 Prototype` | 화면 사이 연결(3장에서 채운다) |
| `99 Archive` | 버린 시안. 지우지 말고 옮긴다 |

## 이름 규칙
- 화면 프레임: `화면번호_화면이름_상태`. 예: `S03_예약_오류`. 검사식 `^S\d{2}_[가-힣A-Za-z]+_[가-힣A-Za-z]+$`.
- 파일 이름에 `최종`, `수정`, `진짜` 같은 말을 넣지 않는다. 판을 나누고 싶으면 버전 기록이나 branch를 쓴다.

## Component 규칙
- 버튼·입력창·목록 셀·상단바 4종을 main component로 만든다. 화면에는 instance만 둔다.
- 화면마다 다른 문구는 instance의 글자 override로 바꾼다. 색을 화면마다 바꿔야 하면 override 대신 variant를 만든다.
- instance를 detach하지 않는다. 꼭 필요하면 `Cover`에 이유와 위치를 적는다.

## Auto layout 규칙
- 모든 화면 프레임과 component에 auto layout을 쓴다.
- 화면 폭을 채울 요소(상단바, 목록 셀, 입력창, 큰 버튼)는 Fill container, 버튼 안 라벨은 Hug contents.
- padding·gap은 `00 Foundations`의 간격 값(4의 배수)만 쓴다.

## 검증 절차 (`checks/ch02.csv`에 기록)
1. 화면 프레임 폭을 375 → 430으로 바꿔 넘침 0개, 좌우 여백이 다른 요소 0개.
2. 버튼 main component의 색을 바꿔 4개 화면의 버튼에 모두 반영되는지, 문구를 바꿔 문구를 override하지 않은 버튼(목록 화면)에 반영되는지 본다(override한 CTA 문구는 그대로여야 한다).
3. 분리된 instance 0개. Organization 요금제라면 Design System Analytics에서 detach 수를 볼 수 있다는 안내가 도움말에 있다. 그 밖의 요금제에서 확인하는 방법은 `[확인 필요]`.
4. 이름 규칙 위반 프레임 0개.

이 저장소의 `productivity_ch02_project.lint()`가 같은 절차를 합성 파일에 대해 자동으로 돌린다.

## 근거
- Guide to auto layout: https://help.figma.com/hc/en-us/articles/360040451373
- Edit main components: https://help.figma.com/hc/en-us/articles/360038665934
- Apply changes to instances: https://help.figma.com/hc/en-us/articles/360039150733
- Detach an instance from the component: https://help.figma.com/hc/en-us/articles/360038665754
