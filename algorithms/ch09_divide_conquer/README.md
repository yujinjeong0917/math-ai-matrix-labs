# 9장 분할정복 곱셈

<!-- sync:start -->
> [그림으로 익히는 알고리즘](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/algorithms) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [9장 분할정복 곱셈](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/09-divide-conquer.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q algorithms/ch09_divide_conquer
```

| 파일 | 하는 일 |
|---|---|
| `algorithms_ch09_mul.py` | 알고리즘 9장 최소 구현: 학교식 곱셈, Karatsuba 곱셈(두 가지 셈법), 재귀 FFT와 FFT 곱셈. |
| `experiments.py` | 알고리즘 9장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch09_divide_conquer/experiments.py` 로 실행한다. |
<!-- sync:end -->
