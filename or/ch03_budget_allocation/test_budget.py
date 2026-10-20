import pytest
from budget import analytic_solution, grid_solution, revenue

def test_analytic_and_grid_agree():
    x_a, x_b = analytic_solution()
    gx_a, gx_b, value = grid_solution()
    assert x_a == pytest.approx(207 / 57)
    assert gx_a == pytest.approx(x_a, abs=0.001)
    assert gx_b == pytest.approx(x_b, abs=0.001)
    assert value == pytest.approx(33.042369, abs=1e-5)

def test_interior_beats_simple_allocations():
    x_a, _ = analytic_solution()
    assert revenue(x_a) > max(revenue(0), revenue(3), revenue(6))

def test_infeasible_allocation_fails():
    with pytest.raises(ValueError):
        revenue(7)
