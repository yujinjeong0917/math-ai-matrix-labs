# math-ai-matrix-labs

[math-ai-matrix](../math-ai-matrix)의 학습 트랙(`tracks/`) 실습 저장소다. 장마다 NumPy 최소 구현, 같은 입력에서 수치를 대조하는 PyTorch 구현, 비교 실험, 테스트가 한 폴더에 있다.

## 실행

```bash
uv sync                      # Python 3.12, numpy 2.3.3, torch 2.8.0 (uv.lock 고정)
uv run pytest -q             # 모든 장의 테스트
uv run python llm/ch07_self_attention/experiments.py   # 7장 비교 실험, results/ch07.json 갱신
```

## 구성

| 폴더 | 웹 챕터 | 내용 |
|---|---|---|
| `llm/ch07_self_attention/` | `math-ai-matrix/tracks/llm/07-self-attention.html` | RNN vs self-attention: 거리별 기울기, 길이별 계산량, 학습 안정성, √d 스케일 |

장마다 파일 역할은 같다.

- `attention_np.py`: NumPy 최소 구현. 원리가 드러나는 가장 작은 코드.
- `attention_torch.py`: PyTorch 대응 구현과 학습용 모델.
- `experiments.py`: 비교 실험. 결과는 `results/*.json`에 저장하고, 웹 챕터는 그 숫자를 그대로 인용한다.
- `test_*.py`: NumPy↔PyTorch 수치 일치, 마스크, 수식 성질 검증.

데이터는 모두 합성 데이터라 별도 라이선스가 없다. 실측 시간은 기기마다 다르지만 FLOP 수와 테스트 결과는 같아야 한다.
