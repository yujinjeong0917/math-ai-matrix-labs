# 5장 순환망과 LSTM

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [5장 순환망과 LSTM](https://math-ai-matrix-pi.vercel.app/tracks/llm/05-rnn-lstm.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch05_rnn_lstm
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 5장 비교 실험. `uv run python llm/ch05_rnn_lstm/experiments.py` 로 실행한다 (단일 스레드 CPU에서 약 5분). |
| `rnn_np.py` | 5장 NumPy 최소 구현: 순환망(RNN)과 LSTM, 손으로 쓴 시간 역전파(BPTT), 기울기 클리핑. |
| `rnn_torch.py` | 5장 PyTorch 대응 구현. |
<!-- sync:end -->
