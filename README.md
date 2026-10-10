# math-ai-matrix-labs

[수학 × AI 매트릭스](https://math-ai-matrix-pi.vercel.app) 학습 트랙의 실습 코드예요. 장마다 원리가 드러나는 NumPy 최소 구현, 같은 입력에서 수치를 대조하는 PyTorch 구현, 비교 실험, 테스트가 한 폴더에 있어요. 웹 챕터의 실험 숫자는 모두 각 폴더의 `results/*.json`에서 그대로 옮겼어요.

## 실행

```bash
uv sync                      # Python 3.12, numpy 2.3.3, torch 2.8.0 (uv.lock으로 고정)
uv run pytest -q             # 모든 장의 테스트
uv run python llm/ch07_self_attention/experiments.py   # 예: 7장 비교 실험, results/ch07.json 갱신
```

## 장 목록

| 폴더 | 장 | 웹 |
|---|---|---|
| `llm/ch01_bigram/` | 밑바닥부터 만드는 LLM 1장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/01-bigram.html) |
| `llm/ch02_ngram/` | 밑바닥부터 만드는 LLM 2장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/02-ngram.html) |
| `llm/ch03_tokenizer/` | 밑바닥부터 만드는 LLM 3장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/03-tokenizer.html) |
| `llm/ch04_embedding_mlp/` | 밑바닥부터 만드는 LLM 4장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/04-embedding-mlp.html) |
| `llm/ch05_rnn_lstm/` | 밑바닥부터 만드는 LLM 5장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/05-rnn-lstm.html) |
| `llm/ch06_attention_seq2seq/` | 밑바닥부터 만드는 LLM 6장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/06-attention-seq2seq.html) |
| `llm/ch07_self_attention/` | 밑바닥부터 만드는 LLM 7장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/07-self-attention.html) |
| `llm/ch08_positional/` | 밑바닥부터 만드는 LLM 8장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/08-positional.html) |
| `llm/ch09_residual_norm/` | 밑바닥부터 만드는 LLM 9장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/09-residual-norm.html) |
| `llm/ch10_train_gpt/` | 밑바닥부터 만드는 LLM 10장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/10-train-gpt.html) |
| `llm/ch11_decoding/` | 밑바닥부터 만드는 LLM 11장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/11-decoding.html) |
| `llm/ch12_kv_cache/` | 밑바닥부터 만드는 LLM 12장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/12-kv-cache.html) |
| `llm/ch13_scaling/` | 밑바닥부터 만드는 LLM 13장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/13-scaling.html) |
| `rl/ch00_map/` | 밑바닥부터 만드는 강화학습 0장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/00-map.html) |
| `rl/ch01_bandit/` | 밑바닥부터 만드는 강화학습 1장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/01-bandit.html) |
| `rl/ch02_mdp_bellman/` | 밑바닥부터 만드는 강화학습 2장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/02-mdp-bellman.html) |
| `rl/ch03_policy_evaluation/` | 밑바닥부터 만드는 강화학습 3장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/03-policy-evaluation.html) |
| `rl/ch04_policy_value_iteration/` | 밑바닥부터 만드는 강화학습 4장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/04-policy-value-iteration.html) |
| `rl/ch05_monte_carlo/` | 밑바닥부터 만드는 강화학습 5장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/05-monte-carlo.html) |
| `rl/ch06_td_sarsa_qlearning/` | 밑바닥부터 만드는 강화학습 6장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/06-td-sarsa-qlearning.html) |
| `rl/ch07_nstep_td_lambda/` | 밑바닥부터 만드는 강화학습 7장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/07-nstep-td-lambda.html) |
| `rl/ch08_function_approx/` | 밑바닥부터 만드는 강화학습 8장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/08-function-approx.html) |
| `rl/ch09_dqn/` | 밑바닥부터 만드는 강화학습 9장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/09-dqn.html) |
| `rl/ch10_policy_gradient/` | 밑바닥부터 만드는 강화학습 10장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/10-policy-gradient.html) |
| `rl/ch11_trpo_ppo/` | 밑바닥부터 만드는 강화학습 11장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/11-trpo-ppo.html) |
| `rl/chbridge_rlhf_dpo/` | 밑바닥부터 만드는 강화학습 연결 장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/bridge-rlhf-dpo.html) |
| `eval/ch01_mae_rmse/` | 모델 평가 1장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/eval/01-mae-rmse.html) |
| `eval/ch02_rmse_mae_ratio/` | 모델 평가 2장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/eval/02-rmse-mae-ratio.html) |
| `eval/ch03_r_squared/` | 모델 평가 3장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/eval/03-r-squared.html) |
| `eval/ch04_mape/` | 모델 평가 4장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/eval/04-mape.html) |
| `eval/ch05_mase/` | 모델 평가 5장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/eval/05-mase.html) |
| `eval/ch06_normalized_bias/` | 모델 평가 6장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/eval/06-normalized-bias.html) |
| `eval/ch07_casebook/` | 모델 평가 7장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/eval/07-casebook.html) |
| `eval/ch08_classification/` | 모델 평가 8장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/eval/08-classification.html) |
| `eval/ch09_clustering/` | 모델 평가 9장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/eval/09-clustering.html) |
| `algorithms/ch01_complexity/` | 그림으로 익히는 알고리즘 1장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/01-complexity.html) |
| `algorithms/ch02_binary_search/` | 그림으로 익히는 알고리즘 2장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/02-binary-search.html) |
| `algorithms/ch03_sorting_bound/` | 그림으로 익히는 알고리즘 3장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/03-sorting-bound.html) |
| `algorithms/ch04_hashing/` | 그림으로 익히는 알고리즘 4장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/04-hashing.html) |
| `algorithms/ch05_balanced_tree/` | 그림으로 익히는 알고리즘 5장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/05-balanced-tree.html) |
| `algorithms/ch06_graph_traversal/` | 그림으로 익히는 알고리즘 6장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/06-graph-traversal.html) |
| `algorithms/ch07_shortest_path/` | 그림으로 익히는 알고리즘 7장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/07-shortest-path.html) |
| `algorithms/ch08_greedy/` | 그림으로 익히는 알고리즘 8장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/08-greedy.html) |
| `algorithms/ch09_divide_conquer/` | 그림으로 익히는 알고리즘 9장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/09-divide-conquer.html) |
| `algorithms/ch10_dynamic_programming/` | 그림으로 익히는 알고리즘 10장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/10-dynamic-programming.html) |
| `pipelines/ch01_reproduce/` | ML 파이프라인 구축 1장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/01-reproduce.html) |
| `pipelines/ch02_hidden_debt/` | ML 파이프라인 구축 2장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/02-hidden-debt.html) |
| `pipelines/ch03_tracking/` | ML 파이프라인 구축 3장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/03-tracking.html) |
| `pipelines/ch04_container/` | ML 파이프라인 구축 4장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/04-container.html) |
| `pipelines/ch05_jenkins_ci/` | ML 파이프라인 구축 5장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/05-jenkins-ci.html) |
| `pipelines/ch06_release/` | ML 파이프라인 구축 6장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/06-release.html) |
| `pipelines/ch07_kfp/` | ML 파이프라인 구축 7장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/07-kfp.html) |
| `pipelines/ch08_lineage/` | ML 파이프라인 구축 8장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/08-lineage.html) |
| `pipelines/ch09_drift/` | ML 파이프라인 구축 9장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/09-drift.html) |
| `productivity/ch01_source_of_truth/` | AX 에세이 1장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/productivity/01-source-of-truth.html) |
| `productivity/ch02_figma_structure/` | AX 에세이 2장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/productivity/02-figma-structure.html) |
| `productivity/ch03_handoff/` | AX 에세이 3장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/productivity/03-handoff.html) |
| `productivity/ch04_notion_relations/` | AX 에세이 4장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/productivity/04-notion-relations.html) |
| `productivity/ch05_sheets/` | AX 에세이 5장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/productivity/05-sheets.html) |
| `productivity/ch06_connect/` | AX 에세이 6장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/productivity/06-connect.html) |

AI 도구 실전 트랙은 작성 중이에요. 전체 목록은 [학습 트랙](https://math-ai-matrix-pi.vercel.app/tracks/)에서 볼 수 있어요.

## 폴더마다 같은 구성

- `*_np.py`: NumPy 최소 구현. 원리가 드러나는 가장 작은 코드예요.
- `*_torch.py`: PyTorch 대응 구현과 학습용 모델(필요한 장만).
- `experiments.py`: 비교 실험. 결과는 `results/*.json`에 저장해요.
- `test_*.py`: NumPy와 PyTorch의 수치 일치, 수식 성질 검증.

데이터는 모두 코드 안에서 만드는 합성 데이터예요. 실측 시간은 기기마다 다르지만, 계산량과 테스트 결과는 같아야 해요.
