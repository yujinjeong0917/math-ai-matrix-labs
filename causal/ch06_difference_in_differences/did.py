"""6장 실습: 이중차분(DiD, Difference-in-Differences)의 2x2 손계산과 포화 회귀 계수.

무엇을 재는가: 처치군과 대조군의 전후 평균 네 칸만으로 효과(처치군 변화 - 대조군 변화)를 구하고,
같은 값이 회귀 Y = b0 + b1*treated + b2*after + b3*treated*after 의 상호작용 계수 b3와 같음을 보인다.
입력: 네 칸의 평균(처치 전/후, 대조 전/후; 예시 수치, 가상). 출력: 효과 하나 또는 계수 dict.
테스트의 기대값(효과 10; 계수 80, 20, 20, 10)이다(장 본문 숫자와의 대응은 README로 확인 필요).
핵심 가정(평행 추세): 처치가 없었다면 두 집단의 변화가 같았을 것이다. 이 코드는 그 가정을 검증하지 않는다.
"""


def did(treated_before: float, treated_after: float, control_before: float, control_after: float) -> float:
    """처치군 변화에서 대조군 변화(시간만의 변화로 간주)를 뺀 이중차분 효과."""
    return (treated_after - treated_before) - (control_after - control_before)

def saturated_coefficients(treated_before: float, treated_after: float, control_before: float, control_after: float) -> dict[str, float]:
    # Y = beta0 + beta1*treated + beta2*after + beta3*treated*after
    # 칸 4개에 계수 4개라 회귀가 네 평균을 정확히 재현한다(포화 모형). 각 계수는 칸 평균의 차로 바로 풀린다
    beta0 = control_before
    beta1 = treated_before - control_before  # 처치 전부터 있던 집단 간 수준 차이
    beta2 = control_after - control_before  # 대조군에서 본 시간 변화(공통 추세로 가정)
    beta3 = did(treated_before, treated_after, control_before, control_after)
    # 상호작용 계수 beta3가 곧 이중차분 효과
    return {"intercept": beta0, "treated": beta1, "after": beta2, "interaction": beta3}

def predict(coefficients: dict[str, float], treated: int, after: int) -> float:
    """treated, after가 각각 0 또는 1인 칸의 회귀 예측값."""
    return (coefficients["intercept"] + coefficients["treated"] * treated
            + coefficients["after"] * after + coefficients["interaction"] * treated * after)
