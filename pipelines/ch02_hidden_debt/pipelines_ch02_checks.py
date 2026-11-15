"""2장 검사: 데이터 계약(스키마), 학습·서빙 분포 비교, ML Test Score 계산.

SCHEMA의 범위는 학습 데이터 통계가 아니라 도메인 지식으로 정했다.
Breck 등(2017)의 Data 1 설명처럼 "기대를 먼저 적고 데이터와 비교"하는 쪽이다.
"""

import numpy as np

SCHEMA = {
    "tenure_months": {"kind": "numeric", "min": 1, "max": 120},
    "monthly_fee": {"kind": "numeric", "min": 10_000, "max": 300_000, "unit": "KRW"},
    "support_calls": {"kind": "numeric", "min": 0, "max": 50},
    # 744 = 31일 x 24시간
    "usage_hours": {"kind": "numeric", "min": 0, "max": 744},  # 한 달은 최대 744시간
    "age": {"kind": "numeric", "min": 14, "max": 100},
    "discount_rate": {"kind": "numeric", "min": 0.0, "max": 1.0},
    "plan": {"kind": "code", "allowed": {0, 1, 2}},
    "region": {"kind": "code", "allowed": {0, 1, 2, 3}},
}
# 검사 임계값은 모두 모형의 선택이다. 값에 따라 오탐·미탐이 달라진다(experiments.py E3b, E4)
MAX_MISSING = 0.01  # 빈 값 비율 상한
MAX_OUT_OF_RANGE = 0.01  # 범위 밖 비율 상한(가끔 있는 이상치는 봐준다)


def validate(batch, schema=SCHEMA):
    """계약 위반 목록을 돌려준다. 빈 목록이면 통과."""
    errors = []
    for col, rule in schema.items():
        if col not in batch:
            errors.append(f"{col}: 열이 없음")
            continue
        v = np.asarray(batch[col])
        if rule["kind"] == "numeric":
            if not np.issubdtype(v.dtype, np.number):
                errors.append(f"{col}: 숫자가 아님({v.dtype})")
                continue
            # 빈 값(NaN) 비율. 정수 코드 열은 NaN을 담을 수 없어 수치 열에서만 센다
            miss = float(np.mean(np.isnan(v)))
            if miss > MAX_MISSING:
                errors.append(f"{col}: 빈 값 {miss:.1%}")
            ok = v[~np.isnan(v)]
            # 범위 밖 비율은 NaN을 뺀 값 기준. 상한을 넘으면 오류
            out = float(np.mean((ok < rule["min"]) | (ok > rule["max"])))
            if out > MAX_OUT_OF_RANGE:
                errors.append(f"{col}: 범위 [{rule['min']}, {rule['max']}] 밖 {out:.1%}")
        else:
            if not np.issubdtype(v.dtype, np.integer):
                errors.append(f"{col}: 정수 코드가 아님({v.dtype})")
                continue
            # 허용 코드 집합 밖의 값. 코드 순서만 바뀐 경우(0,1,2 안에서 뒤섞임)는 여기서 잡지 못한다
            bad = set(np.unique(v).tolist()) - rule["allowed"]
            if bad:
                errors.append(f"{col}: 허용되지 않은 코드 {sorted(bad)}")
    extra = set(batch) - set(schema)
    if extra:
        errors.append(f"계약에 없는 열 {sorted(extra)}")
    return errors


def profile(batch, schema=SCHEMA):
    """학습 표의 요약: 수치 열은 평균·표준편차, 코드 열은 코드별 비율."""
    out = {}
    for col, rule in schema.items():
        v = np.asarray(batch[col])
        if rule["kind"] == "numeric":
            out[col] = {"mean": float(v.mean()), "std": float(v.std())}
        else:
            levels = sorted(rule["allowed"])
            out[col] = {"freq": [float(np.mean(v == c)) for c in levels]}
    return out


# 평균 이동은 학습 표준편차 단위로 잰다: |평균_서빙 - 평균_학습| / 표준편차_학습
MEAN_SHIFT_LIMIT = 0.25  # 평균이 학습 표준편차의 0.25배보다 많이 움직이면 경보
TVD_LIMIT = 0.10  # 코드 비율의 총변동거리 0.5·Σ|p-q|가 0.1보다 크면 경보


def drift_alerts(train_profile, batch, schema=SCHEMA):
    """학습 때 요약과 서빙 표를 비교한다. Breck 등 Monitor 3의 '분포 통계 비교' 방식."""
    alerts = {}
    now = profile(batch, schema)
    for col, ref in train_profile.items():
        if "mean" in ref:
            # ref["std"]가 0인 열(상수 열)이면 0으로 나눈다. 이 장 데이터에는 해당 열이 없다
            shift = abs(now[col]["mean"] - ref["mean"]) / ref["std"]
            if shift > MEAN_SHIFT_LIMIT:
                alerts[col] = round(shift, 3)
        else:
            # 총변동거리 TVD = 0.5 * sum|p - q|, 0(같은 분포)~1(완전히 다름)
            tvd = 0.5 * float(np.sum(np.abs(np.array(now[col]["freq"]) - np.array(ref["freq"]))))
            if tvd > TVD_LIMIT:
                alerts[col] = round(tvd, 3)
    return alerts


# ---- ML Test Score (Breck 등 2017, §VI.A) ----
SECTIONS = ("data", "model", "infra", "monitor")
POINTS = {"none": 0.0, "manual": 0.5, "automated": 1.0}


def ml_test_score(items):
    """items: [(section, 시험 이름, 'none'|'manual'|'automated'), ...]

    수동으로 실행하고 결과를 문서로 남기면 0.5점, 반복 자동 실행 체계가 있으면 1점.
    네 영역을 각각 합한 뒤, 최종 점수는 그 가운데 최솟값이다.
    """
    # 영역별 합을 구한 뒤 최솟값이 최종 점수. 한 영역이 0이면 다른 영역이 높아도 0
    sums = {s: 0.0 for s in SECTIONS}
    for section, _, status in items:
        sums[section] += POINTS[status]
    return {"by_section": sums, "final": min(sums.values())}


# 이 장의 저장소가 실제로 다루는 항목. 아직 CI가 없어서(5장) 모두 '사람이 pytest를 돌리는' 수동 단계다.
THIS_CHAPTER = [
    ("data", "Data 1 · 기대를 스키마로 적었다", "manual"),
    ("data", "Data 7 · 입력 피처 코드를 테스트했다", "manual"),
    ("model", "Model 5 · 더 단순한 모델보다 낫다(여기서는 다수 클래스 기준선만 비교)", "manual"),
    ("model", "Model 6 · 중요한 조각(요금제별)에서도 품질이 충분하다", "manual"),
    ("infra", "Infra 1 · 학습이 재현된다", "manual"),
    ("infra", "Infra 2 · 모델 명세를 단위 테스트했다(NumPy·PyTorch 대조)", "manual"),
    ("infra", "Infra 4 · 서빙 전에 품질을 확인한다", "manual"),
    ("monitor", "Monitor 2 · 입력이 불변식을 지킨다", "manual"),
    ("monitor", "Monitor 3 · 학습과 서빙 분포를 비교한다(분포 통계만, 같은 예제 대조는 없음)", "manual"),
]
