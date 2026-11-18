# 1장 · 화면을 사각형으로 쪼개면 무엇이 남나

<!-- sync:start -->
> [디자이너의 시선으로 레퍼런스 읽기](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/reference-study) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [화면을 큰 사각형으로 쪼개면 어떤 정렬선을 찾을 수 있을까요?](https://math-ai-matrix-pi.vercel.app/tracks/reference-study/01-layout-lines.html)

```bash
cd reference-study/ch01_layout_lines
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `layout_lines.py` | 1장: 화면을 상자로 쪼개고 정렬선을 찾는다. |
<!-- sync:end -->

홈의 "시작하기" 구역과 첫 화면을 상자로 나누고, 정렬선(왼쪽·중앙·위)을 찾습니다. x 좌표와 너비는 CSS(`.wrap`, `.group`, `.grid--2`, 419px 이하에서 `.wrap`의 좌우 여백 16px 규칙)에서 계산한 값이고, y 좌표·높이·글자 상자 너비는 가상, 예시용입니다. 파이썬 표준 라이브러리만 씁니다.

## 실행

```bash
python3 layout_lines.py
python3 -m unittest -q
```

## 기대 출력

```text
[폭 390px] .wrap--wide 안쪽 너비 358px
[폭 1024px] .wrap--wide 안쪽 너비 980px
[폭 1440px] .wrap--wide 안쪽 너비 1156px
[폭 1440px] 머리 열 x=142.00 너비 346.34 / 본문 열 x=528.34 너비 769.66
열 수 식 값 (C+g)/(m+g) = 2.207 -> 바닥 2열, 상자 너비 376.83px
[폭 1440px] 가운데 정렬 여백 120px + 안쪽 여백(padding) 22px = 142px
left [(142.0, ['h2 제목', 'kicker', '설명'])]
right []
top [(200.0, ['kicker', '링크 상자1', '링크 상자2'])]
중앙선(첫 화면) [(720.0, ['h1', 'kicker', 'lead', 'stats', '버튼줄'])]
상자 높이를 모두 2배로 해도 왼쪽, 오른쪽, 위쪽 선이 같은가: True
밀도(가상 y, h) 0.224
```

## 이 실습이 확인하지 못하는 것

- 브라우저에서 실제 상자 좌표를 재지 않았습니다. CSS 값에서 계산한 값이라 줄바꿈과 글꼴 때문에 실제와 다를 수 있어요.
- y 좌표와 높이는 가상이라 밀도 값은 방법을 보여 주는 숫자일 뿐입니다.
- 정렬선이 있다는 것과 그 정렬이 의도였다는 것은 다릅니다.
