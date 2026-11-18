# 0장 · 처음 3초에 무엇이 눈에 들어오나

<!-- sync:start -->
> [디자이너의 시선으로 레퍼런스 읽기](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/reference-study) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [화면을 처음 3초 봤을 때 무엇이 눈에 들어올까요?](https://math-ai-matrix-pi.vercel.app/tracks/reference-study/00-first-impression.html)

```bash
cd reference-study/ch00_first_impression
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `impression.py` | 0장: 처음 3초에 눈에 들어오는 것은 글자 크기 순서와 얼마나 겹치나. |
<!-- sync:end -->

이 사이트 홈 첫 화면의 글자 크기를 `assets/site.css`의 clamp 값으로 계산해 화면 폭마다 크기 순서와 h1÷본문 비율을 냅니다. 화면 폭 390·600·1440px은 가상, 예시용입니다. 파이썬 표준 라이브러리만 씁니다.

## 실행

```bash
python3 impression.py
python3 -m unittest -q
```

## 기대 출력

```text
[폭 390px] h1 34.0px / 리드 19.0px / 본문 17px / h1÷본문 2.000배
  크기 순서: h1 제목(34.0) > 리드 문장(19.0) > 본문·버튼 글자(17.0) > kicker(14.0)
[폭 600px] h1 37.2px / 리드 19.0px / 본문 17px / h1÷본문 2.188배
  크기 순서: h1 제목(37.2) > 리드 문장(19.0) > 본문·버튼 글자(17.0) > kicker(14.0)
[폭 1440px] h1 56.0px / 리드 24.0px / 본문 17px / h1÷본문 3.294배
  크기 순서: h1 제목(56.0) > 리드 문장(24.0) > 본문·버튼 글자(17.0) > kicker(14.0)
h1이 최대 56px에 닿는 폭: 903.2px
h1이 최소 34px에서 벗어나는 폭: 548.4px
리드가 최소 19px에서 벗어나는 폭: 791.7px
리드가 최대 24px에 닿는 폭: 1000.0px
첫 화면 위쪽 여백(.hero padding-top): 96px
```

## 이 실습이 확인하지 못하는 것

- 글자 크기 순서가 "처음 3초에 눈에 들어오는 순서"와 같은지는 재지 않습니다. 사람이 어디를 먼저 보는지는 시선 추적이나 사용자 관찰이 있어야 알 수 있어요.
- 색, 여백, 이미지가 인상에 주는 영향은 계산하지 않습니다.
- CSS 값은 2026-11-18에 연 파일 기준입니다.
