# 1장. 브라우저에 내려온 건 왜 다 보일까 (`ch01_everything_visible`)

웹 챕터: `math-ai-matrix/tracks/webapp/01-everything-visible.html`

관통 프로젝트(직접 만든 작은 "카드뉴스 편집기" 웹앱)를 표준 라이브러리 `http.server`로 띄우고, 화면을 그리려고 브라우저가 받아 가는 파일(HTML·CSS·JS·폰트·이미지·템플릿 JSON)과, 받아 가지 않고 서버에만 남는 것(가짜 AI 키, 이미지 생성 지시문, 원본 고해상도 템플릿)을 나눠 센다.

1. 편집기를 열 때와 두 동작(템플릿 바꾸기, AI 배경 만들기)에서 오가는 요청 목록과 크기, 캐시 규칙을 모은다.
2. `editor/` 아래 모든 파일을 "내려감 / 지도 파일로 내려감 / 서버에만 있음"으로 나누고, 서버에만 둔 값이 받은 바이트 안에 한 번도 없는지 센다.
3. 압축(주석·빈칸 지우기)과 이름 바꾸기가 무엇을 지우고 무엇을 남기는지, 지도 파일(source map)을 함께 내보내면 원본이 통째로 돌아온다는 것을 확인한다.

## 실행
```bash
uv run python webapp/ch01_everything_visible/experiments.py   # results/ch01.json 갱신
uv run pytest -q webapp/ch01_everything_visible
uv run python webapp/ch01_everything_visible/webapp_ch01_build.py    # src/app.js를 고쳤다면 배포본 다시 만들기
```
표준 라이브러리만 쓴다. 서버는 `127.0.0.1`의 빈 포트에 잠깐 띄웠다 끈다. 바깥 네트워크는 쓰지 않는다.

## 브라우저로 직접 보기 (우리 편집기만 대상)
```bash
uv run python webapp/ch01_everything_visible/webapp_ch01_server.py   # http://127.0.0.1:8000/
```
크롬에서 개발자 도구를 먼저 열고 Network 탭의 Disable cache를 켠 뒤 새로고침하면, `results/ch01.json`의 `e1_requests`와 같은 목록을 볼 수 있다. 기록 양식은 `checks/ch01.csv`.

- 폰트 파일 `fonts/card-sans.woff2`는 진짜 폰트가 아니라 84바이트짜리 자리표시다. 브라우저 콘솔에 글꼴을 읽지 못했다는 경고가 뜨고, 글자는 CSS에 적은 대체 글꼴(system-ui)로 그려진다. 정상이다.
- 브라우저의 Size 값은 응답 머리글이 더해지고, 이 서버는 압축하지 않아서 모형의 `bytes`와 조금 다르다. `gzip_bytes`는 "압축해서 보낸다면"의 값(gzip, mtime=0)이다.

## 파일
| 파일 | 역할 |
|---|---|
| `editor/public/` | 브라우저가 받아 가는 공개 폴더(HTML, CSS, 배포용 JS와 지도 파일, 자리표시 폰트, SVG 그림, 템플릿 JSON) |
| `editor/src/app.js` | 주석이 달린 원본 JS. 그대로 배포하지 않는다 |
| `editor/server_only/` | 서버에만 두는 것: 가짜 키(`DEMO_NOT_A_REAL_KEY_0000`), 생성 지시문, 원본 고해상도 템플릿 |
| `webapp_ch01_build.py` | 압축·이름 바꾸기·지도 파일 만들기 흉내 |
| `webapp_ch01_server.py` | 공개 폴더만 내보내는 서버, 폴더 목록 끄기, 캐시 규칙, `/api/generate-image` 흉내 |
| `webapp_ch01_visible.py` | 브라우저가 따라가는 참조를 흉내 낸 요청 모형(127.0.0.1만), 파일 표, 캐시 모형 |
| `experiments.py` | E0~E8 |
| `data/sources_2026-10-10.json` | 브라우저·개발자 도구 동작의 출처 URL과 확인일 |
| `checks/ch01.csv` | 독자 기록 양식(예시 행은 모형 값) |

## 한계
- 요청 모형은 브라우저가 아니다. HTML의 `src`·`href`, CSS의 `url()`, JS가 부르는 템플릿 목록만 따라간다. 실제 브라우저는 쓰지 않는 글꼴을 받지 않을 수 있고, 같은 파일을 캐시에서 꺼내기도 한다.
- 빌드 흉내는 문자열과 주석만 구분하는 작은 토큰 분리기라 정규식 리터럴·템플릿 문자열이 있는 코드는 다루지 못한다. 지도 파일의 `mappings`(줄·칸 대응표)는 비워 두고 `sourcesContent`만 채웠다.
- 이미지 생성은 진짜 AI를 부르지 않고 입력으로 정해지는 무늬를 그린다. 사용량 제한은 사용자 구분 없이 서버 전체 횟수만 센다.
- 모든 키·지시문·템플릿은 실습용으로 만든 가짜 값이다.

확인일 2026-10-10.
