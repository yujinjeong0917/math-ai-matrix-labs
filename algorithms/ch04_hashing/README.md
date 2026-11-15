# 4장 해시 플러딩

<!-- sync:start -->
> [그림으로 익히는 알고리즘](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/algorithms) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [4장 해시 플러딩](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/04-hashing.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q algorithms/ch04_hashing
```

| 파일 | 하는 일 |
|---|---|
| `algorithms_ch04_hash.py` | 알고리즘 4장 최소 구현: 체이닝 해시 테이블, 고정 해시(x mod m), Carter-Wegman(1979)의 범용 해시족 H1. |
| `experiments.py` | 알고리즘 4장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch04_hashing/experiments.py` 로 실행한다. |
<!-- sync:end -->
