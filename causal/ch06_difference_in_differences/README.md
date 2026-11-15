# 대표 6장. 이중차분

<!-- sync:start -->
> [인과추론](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/causal) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [전후 비교에 대조군 하나를 더하면 무엇이 달라질까요?](https://math-ai-matrix-pi.vercel.app/tracks/causal/06-difference-in-differences.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q causal/ch06_difference_in_differences
```

| 파일 | 하는 일 |
|---|---|
| `did.py` | 6장 실습: 이중차분(DiD, Difference-in-Differences)의 2x2 손계산과 포화 회귀 계수. |
<!-- sync:end -->

웹 챕터: https://math-ai-matrix-pi.vercel.app/tracks/causal/06-difference-in-differences.html

네 집단 평균의 손계산과 포화 회귀의 상호작용 계수가 같은지 대조합니다. 값이 같아도 평행추세 가정을 자동으로 증명하지는 않습니다.

```bash
uv run pytest -q causal/ch06_difference_in_differences
```
