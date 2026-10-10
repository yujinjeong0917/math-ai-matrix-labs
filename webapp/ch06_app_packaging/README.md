# 6장. 웹앱을 앱으로 만드는 네 가지 길 (`ch06_app_packaging`)

웹 챕터: `math-ai-matrix/tracks/webapp/06-app-packaging.html`

1장에서 만든 카드뉴스 편집기를 PWA(홈 화면에 추가해 앱처럼 쓰는 웹앱)로 바꾸고, 네 가지 길(PWA, WebView 포장, 크로스플랫폼, 네이티브)을 공식 문서로 확인한 구조로 비교한다.

1. `pwa/public/manifest.json`과 `pwa/public/sw.js`(서비스 워커)를 직접 만들고 한 줄씩 읽는다. `index.html`에는 manifest 연결과 서비스 워커 등록 몇 줄만 더했다.
2. manifest 검사기: 크로미움 계열 설치 조건(MDN·web.dev), 아이콘 파일·크기 대조, start_url과 scope 관계를 본다. 일부러 망가뜨린 manifest 4개로 시험한다.
3. 오프라인 캐시 전략 모형: 서비스 워커 없음 / cache-first / network-first / stale-while-revalidate / 섞어 쓰기(`sw.js` 규칙)를 같은 방문 순서(온라인 → 오프라인 → 배포 → 온라인 → 온라인 → 오프라인)로 돌려 네트워크 요청, 캐시에서 준 수, 옛 버전을 준 수, 실패 수를 센다.
4. 네 가지 길 비교표(`data/paths_2026-10-10.json`)와 조건별 고르기. 순위를 매기지 않고 "이 조건을 모두 만족하는 길"만 돌려준다.

## 실행
```bash
uv run python webapp/ch06_app_packaging/experiments.py   # results/ch06.json 갱신
uv run pytest -q webapp/ch06_app_packaging
uv run python webapp/ch06_app_packaging/webapp_ch06_icons.py    # 아이콘 다시 만들기
```
표준 라이브러리만 쓴다. 서버는 `127.0.0.1`의 빈 포트에 잠깐 띄웠다 끈다. 바깥 네트워크는 쓰지 않는다. Node.js가 있으면 `node --check`로 JS 문법도 본다.

## 브라우저로 직접 보기
```bash
uv run python webapp/ch06_app_packaging/webapp_ch06_server.py   # http://127.0.0.1:8000/
```
- 서비스 워커는 https 이거나 localhost·127.0.0.1 에서만 등록된다. 같은 와이파이의 휴대폰에서 `http://192.168.x.x:8000`으로 열면 등록되지 않는다(검사기의 `https` 항목이 실패하는 것과 같은 이유).
- 크롬 개발자 도구 Application 탭에서 Manifest, Service workers, Cache Storage를 볼 수 있다. Network 탭에서 Offline을 고르고 새로고침하면 편집기가 뜨고, AI 배경 버튼만 실패한다. 기록 양식은 `checks/ch06.csv`.
- `sw.js`를 고쳐 `version`을 바꾸면, 다음 방문에 새 서비스 워커가 설치되고 그다음 방문(탭을 모두 닫은 뒤)부터 맡는다.

## 파일
| 파일 | 역할 |
|---|---|
| `pwa/public/` | PWA로 바꾼 편집기(1장 파일을 복사하고 manifest·서비스 워커·아이콘을 더함, 지도 파일과 그 주석 줄은 뺌) |
| `pwa/public/sw.js` | 서비스 워커. `/* config:start */` 블록(precache 목록과 경로별 전략)을 파이썬 모형이 그대로 읽는다 |
| `webapp_ch06_manifest.py` | manifest 검사기 |
| `webapp_ch06_icons.py` | 아이콘 PNG 만들기·크기 읽기 |
| `webapp_ch06_cache.py` | 오프라인 캐시 전략 모형 |
| `webapp_ch06_paths.py` | 네 가지 길 비교표와 조건별 고르기 |
| `webapp_ch06_server.py` | 공개 폴더만 내보내는 서버(sw.js는 no-cache, API는 흉내) |
| `experiments.py` | E0~E6 |
| `data/manifests/` | 일부러 망가뜨린 manifest 4개 |
| `data/paths_2026-10-10.json`, `data/sources_2026-10-10.json` | 비교표 값과 출처 URL·확인일 |
| `checks/ch06.csv` | 독자 기록 양식(예시 행은 모형 값) |

## 한계
- 캐시 모형은 브라우저가 아니다. `sw.js`가 `{ cache: "reload" }`로 요청해 HTTP 캐시를 건너뛴다고 두었다(이 옵션이 없으면 하루짜리 HTTP 캐시에서 옛 파일이 나올 수 있다). 첫 방문에는 페이지 파일을 받은 뒤 precache를 다시 네트워크에서 받는다, 새 서비스 워커는 다음 방문에 바로 맡는다고 두었다(실제로는 열린 탭이 모두 닫혀야 한다). 응답 시간, 저장 공간 한도, 브라우저가 저장소를 지우는 경우는 다루지 않는다.
- manifest 검사기의 `chromium` 묶음은 문서에 적힌 조건만 본다. 크롬이 아이콘 크기가 틀린 manifest를 실제로 어떻게 다루는지는 확인하지 않았다. Safari(iOS 26)는 manifest가 없어도 홈 화면 웹앱으로 연다.
- 비교표는 "되는지/안 되는지"와 구조만 적는다. 성능이나 개발 난이도에 점수를 매기지 않는다.

확인일 2026-10-10.
