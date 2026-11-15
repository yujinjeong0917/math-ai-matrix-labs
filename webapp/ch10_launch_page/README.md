# 10장. 소개 페이지 출시 점검

<!-- sync:start -->
> [1인 개발 앱 출시 실전](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/webapp) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [10장 소개 페이지 하나면 출시 준비가 끝날까요?](https://math-ai-matrix-pi.vercel.app/tracks/webapp/10-launch-page.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q webapp/ch10_launch_page
```

| 파일 | 하는 일 |
|---|---|
| `launch_page.py` | 출시 페이지 점검 (1인 개발 앱 출시 실전 10장). |
<!-- sync:end -->

웹 챕터: https://math-ai-matrix-pi.vercel.app/tracks/webapp/10-launch-page.html

소개·개인정보·지원·계정 삭제 URL이 공개돼 있고 같은 앱 이름을 쓰는지 검사합니다. 네트워크를 호출하지 않고 제출용 스냅샷 사전을 판정합니다.

```bash
uv run pytest -q webapp/ch10_launch_page
```
