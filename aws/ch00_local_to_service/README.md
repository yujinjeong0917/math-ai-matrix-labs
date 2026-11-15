# 0장 · 로컬 모델이 서비스가 되려면 무엇이 더 필요한가

<!-- sync:start -->
> [AWS 배포·운영](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/aws) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [내 노트북에서 잘 돌던 모델이 서비스가 되려면 무엇이 더 필요할까요?](https://math-ai-matrix-pi.vercel.app/tracks/aws/00-local-to-service.html)

```bash
cd aws/ch00_local_to_service
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `service_gap.py` | 로컬 모델이 서비스가 되기 전에 비어 있는 항목을 찾는다. AWS 호출 없음, 표준 라이브러리만 쓴다. |
<!-- sync:end -->

노트북에서 돌던 모델에 하루 20건과 2만 건이 들어올 때 얼마나 많은 작업이 겹치는지 재고, 서비스로 내놓기 전에 답해야 할 여덟 물음 중 비어 있는 것을 찾습니다. AWS 계정과 네트워크는 쓰지 않아요.

## 실행

```bash
python3 service_gap.py notebook_state.json
python3 -m unittest -q
```

## 기대 출력

```text
[하루 20건] 도착 간격 4320.00초, 하루 처리 0.1667시간, 평균 동시 작업 0.00694, 최소 작업자 1
[하루 2만 건] 도착 간격 4.32초, 하루 처리 166.6667시간, 평균 동시 작업 6.94444, 최소 작업자 7
노트북 상태: NOT_READY (갖춤 0, 없음 6, 미확인 2)
```

## 재는 것

- 도착 간격 = 86,400초 ÷ 하루 요청 수
- 평균 동시 작업 수 = 하루 요청 수 × 건당 초 ÷ 86,400
- 여덟 물음 중 `False`는 없음, `null`은 아직 확인하지 않음(HOLD)

## 확인하지 못하는 것

- 평균일 뿐이라 요청이 몰리는 순간의 동시 작업 수는 모릅니다. 실제 로그의 최대 동시 요청을 따로 재야 해요.
- 건당 30초는 가상 값입니다. 내 모델에서 직접 재야 해요.
- 어떤 AWS 서비스를 쓸지, 가격이 얼마인지는 이 실습의 범위 밖입니다.
