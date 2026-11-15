"""6장 테스트: pytest 형식의 단순 assert 함수(unittest 클래스가 아님). 이 폴더에서 `pytest`로 실행한다.
손계산 효과, 포화 회귀의 계수와 네 칸 재현, 공통 시간 변화가 상쇄되는 성질을 확인한다."""
from did import did, predict, saturated_coefficients

def test_hand_difference_is_ten():
    # (130 - 100) - (100 - 80) = 30 - 20 = 10
    assert did(100, 130, 80, 100) == 10

def test_saturated_regression_reproduces_four_cells():
    # 절편 80(대조 전), 처치 +20(수준 차이), 시간 +20(대조 변화), 상호작용 +10(DiD 효과)
    coef = saturated_coefficients(100, 130, 80, 100)
    assert coef == {"intercept": 80, "treated": 20, "after": 20, "interaction": 10}
    # 네 칸 평균이 그대로 복원된다
    assert predict(coef, 0, 0) == 80
    assert predict(coef, 0, 1) == 100
    assert predict(coef, 1, 0) == 100
    assert predict(coef, 1, 1) == 130

def test_common_time_change_cancels():
    # 두 집단이 같은 +20 시간 변화를 겪어도 처치군에 +10만 더해지면 효과는 그대로 10(120->150, 100->120)
    assert did(120, 150, 100, 120) == 10
