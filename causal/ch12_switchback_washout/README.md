# 12장. 스위치백과 세척기간

<!-- sync:start -->
> [인과추론](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/causal) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [지역을 나눌 수 없다면 시간대마다 켰다 꺼서 효과를 잴 수 있을까요?](https://math-ai-matrix-pi.vercel.app/tracks/causal/12-switchback-washout.html)

```bash
cd causal/ch12_switchback_washout
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `switchback.py` | 12장 스위치백과 세척기간. 표준 라이브러리만 쓴다. 가상 온라인 마켓: 지역 한 곳의 '오후 타임세일 문구'를 시간대마다 켰다 껐다 한다. |
<!-- sync:end -->

한 지역의 문구를 시간대마다 켰다 끄는 스위치백 실험에서, 이월효과(tau=+6, 1칸 뒤 -3, 2칸 뒤 -1.5)가 있을 때 블록 길이와 세척기간(블록이 바뀐 직후 버리는 슬롯 수)에 따라 추정값이 어떻게 달라지는지 300번 반복해 잽니다. 진짜 정상상태 효과는 6-3-1.5 = 1.5입니다.

## 실행

```bash
cd causal/ch12_switchback_washout
python3 switchback.py          # 숫자를 출력해요
python3 -m unittest -q  # 테스트 (pytest도 됩니다)
```

파이썬 표준 라이브러리만 쓰고 시드를 고정해서 같은 출력이 나옵니다.

## 기대 출력 (`python3 switchback.py`)

```text
true steady 1.5
block= 1 washout=0 {'mean': 6.05, 'sd': 0.74, 'bias': 4.55}
block= 2 washout=0 {'mean': 4.41, 'sd': 1.09, 'bias': 2.91}
block= 4 washout=0 {'mean': 3.01, 'sd': 1.21, 'bias': 1.51}
block= 4 washout=1 {'mean': 1.92, 'sd': 1.17, 'bias': 0.42}
block= 4 washout=2 {'mean': 1.47, 'sd': 1.17, 'bias': -0.03}
block= 8 washout=2 {'mean': 1.59, 'sd': 1.03, 'bias': 0.09}
block=12 washout=2 {'mean': 1.46, 'sd': 0.93, 'bias': -0.04}
block=24 washout=2 {'mean': 1.55, 'sd': 1.03, 'bias': 0.05}
block=60 washout=2 {'mean': 1.52, 'sd': 0.81, 'bias': 0.02}
kept_share block= 4 washout=1 0.871
kept_share block= 4 washout=2 0.742
kept_share block= 8 washout=2 0.868
kept_share block=24 washout=2 0.951
kept_share block=60 washout=2 0.975
flat_base_sd block= 1 washout=0 0.46
flat_base_sd block= 4 washout=2 0.44
flat_base_sd block=12 washout=2 0.42
flat_base_sd block=24 washout=2 0.41
flat_base_sd block=60 washout=2 0.4
```

## 이 실습이 확인하지 못하는 것

이월효과가 정확히 2칸이고 그 길이를 안다고 가정한 장난감입니다. 실제로는 이월 길이를 모릅니다. 요일·시간대 패턴이 약해서 긴 블록의 계절성 위험은 드러나지 않습니다. 시간대 사이에 사용자가 겹치는 문제와 표준오차는 다루지 않습니다.
