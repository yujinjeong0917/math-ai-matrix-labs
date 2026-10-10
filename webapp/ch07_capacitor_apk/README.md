# 7장. Capacitor로 APK 만들기: 스토어 없이 배포까지 (`ch07_capacitor_apk`)

웹 챕터: `math-ai-matrix/tracks/webapp/07-capacitor-apk.html`

6장에서 PWA로 만든 카드뉴스 편집기를 Capacitor로 Android 앱(APK)에 담는다고 해 보고, 그때 쓰는 설정 파일을 한 줄씩 읽고 점검한다. 이어서 가상의 행사장 주문 앱을 태블릿 8대에 스토어 없이 나눠 주는 하루를 모형으로 돌려, 서명 키와 versionCode가 업데이트를 어떻게 정하는지 센다.

1. `capacitor.config.json`, `AndroidManifest.xml`, `build.gradle`, `variables.gradle`의 바른 판과 잘못 고친 판을 점검기로 훑는다(appId 형식, server.url·cleartext, WebView 디버깅, 권한 최소화, minSdk, 비밀번호·키 파일 위치).
2. 서명 키 관리 점검표 10개 항목 중 기계 항목 6개를 두 판에 채우고, 사람 항목 4개는 `checks/ch07.csv`에 적는다.
3. versionCode 예시 규칙(큰 변경 × 10000 + 중간 변경 × 100 + 작은 수정)과 설치·업데이트 모형으로 "같은 키·더 큰 번호만 업데이트된다"를 확인한다.

**실제 APK는 만들지 않는다.** Android SDK, Java, Node 없이 돈다. 진짜 빌드 명령은 웹 챕터에 보여만 준다.

## 실행
```bash
uv run python webapp/ch07_capacitor_apk/experiments.py                                     # results/ch07.json 갱신
uv run pytest -q webapp/ch07_capacitor_apk
uv run python webapp/ch07_capacitor_apk/webapp_ch07_config.py webapp/ch07_capacitor_apk/project/cardnews_mistake   # 막음이 있으면 종료 코드 1
```
내 Capacitor 프로젝트를 점검하려면 위 여섯 파일(`capacitor.config.json`, `AndroidManifest.xml`, `build.gradle`, `variables.gradle`, `app_features.json`, `gitignore.txt`)을 한 폴더에 복사해 넣고 돌린다. `capacitor.config.ts`는 읽지 못하니 JSON으로 옮겨 적는다.

## 가짜 값 규칙
- appId는 실제로 없는 도메인 `.invalid`를 뒤집은 `invalid.demo.cardnews`, `invalid.demo.eventorder`다.
- 비밀번호는 `DEMO_NOT_A_REAL_PASSWORD`, 키 파일 `release.jks`는 글자 한 줄짜리 자리표시, 인증서는 이름표 문자열의 SHA-256(`DEMO-…`)이다. `test_no_real_secrets`가 확인한다.
- `server.url`의 `192.168.0.10`은 사설망 예시 주소다.
- `.gitignore`는 실습 저장소에서 보이게 `gitignore.txt`로 이름을 바꿨다.
- 행사장 주문 앱은 "이런 앱을 만든다고 해 볼게요" 수준의 가상 사례다. 태블릿 대수·기기 API 수준·단계는 이 장에서 정한 값이다.

## 파일
| 파일 | 역할 |
|---|---|
| `project/cardnews_good/` | 바른 판: 설정 6개 + `keystore.properties.example` |
| `project/cardnews_mistake/` | 잘못 고친 판: 개발 설정이 남고, 권한을 "혹시 몰라" 넣고, 비밀번호·키 파일이 저장소에 있는 판 |
| `webapp_ch07_config.py` | 설정 점검기(규칙마다 출처 id) |
| `webapp_ch07_signing.py` | 서명 키 관리 점검표, keytool 유효기간 계산, 배포 파일 SHA-256 |
| `webapp_ch07_update.py` | 설치·업데이트 모형(minSdk, 서명 인증서, versionCode, 알 수 없는 앱 설치 허용), 행사 하루 시나리오, "무엇을 바꾸면 새 APK가 필요한가" 표 |
| `experiments.py` | E0~E7 |
| `data/sources_2026-10-10.json` | 사실 36개의 출처 URL·확인일, 확인하지 못한 것 4개 |
| `checks/ch07.csv` | 서명 점검표 기록 양식(예시 행은 모형 값) |

## 한계
- 모형은 Android가 아니다. 문서에 적힌 네 규칙만 흉내 내고, 같은 versionCode로 다시 설치하는 경우는 기기 동작을 확인하지 못해 `unknown_same_code`로 둔다.
- 삭제하면 앱 데이터가 지워진다는 건 이 장의 가정([해석])이다.
- 점검기는 아는 규칙만 본다. 0건은 "이 규칙들에 걸리지 않았다"는 뜻이다. ProGuard·네트워크 보안 설정 파일·딥링크·백업 규칙은 보지 않는다.
- 라이브 업데이트는 Capacitor 문서(회사 설명)와 Play 정책으로만 확인했다(`not_confirmed`).

확인일 2026-10-10.
