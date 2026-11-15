# 연결 장 RLHF·DPO

<!-- sync:start -->
> [밑바닥부터 만드는 강화학습](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/rl) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [연결 장 RLHF·DPO](https://math-ai-matrix-pi.vercel.app/tracks/rl/bridge-rlhf-dpo.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q rl/chbridge_rlhf_dpo
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 강화학습 연결 장 실험. `cd math-ai-matrix-labs && uv run python rl/chbridge_rlhf_dpo/experiments.py` |
| `rl_chbridge_rlhf_dpo_np.py` | 강화학습 연결 장 NumPy 최소 구현: 비교 데이터로 보상 모델 만들기(Bradley-Terry), KL 정규 정책경사(RLHF), |
| `rl_chbridge_rlhf_dpo_torch.py` | 강화학습 연결 장 PyTorch 대응 구현. 손으로 쓴 NumPy 기울기를 autograd와 대조하고, |
| `viz.py` | results/chbridge.json과 logs/chbridge_samples.jsonl을 터미널에 그린다(외부 패키지 없음). |
<!-- sync:end -->
