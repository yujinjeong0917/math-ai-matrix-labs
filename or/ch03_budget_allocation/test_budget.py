import pytest
from budget import analytic_solution, grid_solution, revenue

# 주의: pytest를 쓰므로 `python3 -m unittest -q`가 아니라 `uv run pytest -q or/ch03_budget_allocation`로 돌려야 한다
def test_analytic_and_grid_agree():
    x_a, x_b = analytic_solution()
    gx_a, gx_b, value = grid_solution()
    assert x_a == pytest.approx(207 / 57)
    # 격자 간격이 0.001이라 최적점과 최대 0.0005 어긋날 수 있어 허용 오차를 간격과 같게 둔다
    assert gx_a == pytest.approx(x_a, abs=0.001)
    assert gx_b == pytest.approx(x_b, abs=0.001)
    # 33.042369는 해석해를 revenue에 넣은 값(소수 6자리 반올림)
    assert value == pytest.approx(33.042369, abs=1e-5)

def test_interior_beats_simple_allocations():
    # 한 채널에 몰기(0, 6)나 반반(3)보다 내부 해가 낫다: 포화 때문에 분산 배분이 유리
    x_a, _ = analytic_solution()
    assert revenue(x_a) > max(revenue(0), revenue(3), revenue(6))

def test_infeasible_allocation_fails():
    with pytest.raises(ValueError):
        revenue(7)   # 예산 6을 넘는 x_a=7이면 x_b가 음수
