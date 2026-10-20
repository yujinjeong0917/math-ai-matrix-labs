from did import did, predict, saturated_coefficients

def test_hand_difference_is_ten():
    assert did(100, 130, 80, 100) == 10

def test_saturated_regression_reproduces_four_cells():
    coef = saturated_coefficients(100, 130, 80, 100)
    assert coef == {"intercept": 80, "treated": 20, "after": 20, "interaction": 10}
    assert predict(coef, 0, 0) == 80
    assert predict(coef, 0, 1) == 100
    assert predict(coef, 1, 0) == 100
    assert predict(coef, 1, 1) == 130

def test_common_time_change_cancels():
    assert did(120, 150, 100, 120) == 10
