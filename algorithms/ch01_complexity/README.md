# 1장 빠르다는 것

<!-- sync:start -->
> [그림으로 익히는 알고리즘](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/algorithms) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [1장 빠르다는 것](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/01-complexity.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q algorithms/ch01_complexity
```

| 파일 | 하는 일 |
|---|---|
| `algorithms_ch01_dup.py` | 알고리즘 1장 최소 구현: 리스트에 같은 값이 두 번 이상 있는지 세 가지 방법으로 찾는다. |
| `algorithms_ch01_timing.py` | 알고리즘 1장 시간 측정 핵심. 표준 라이브러리만 쓴다(Python 3.9 이상). |
| `experiments.py` | 알고리즘 1장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch01_complexity/experiments.py` 로 실행한다. |
<!-- sync:end -->
