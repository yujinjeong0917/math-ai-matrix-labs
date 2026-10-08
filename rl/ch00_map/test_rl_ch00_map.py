import math
from map_numbers import build, discounted_return


def test_return_example():
    assert math.isclose(discounted_return([1, 1, 10], 0.9), 10.0)


def test_chain_numbers():
    c = build()["chain"]
    assert math.isclose(c["true"]["A"], 0.63) and math.isclose(c["true"]["B"], 0.7)
    assert math.isclose(c["mc"]["A"], 0.7) and math.isclose(c["mc"]["B"], 0.8)
    assert math.isclose(c["td"]["A"], 0.52) and math.isclose(c["td"]["B"], 0.8)
    assert math.isclose(c["dp"]["A"], 0.54) and math.isclose(c["dp"]["B"], 0.7)


def test_sarsa_vs_q():
    s = build()["sarsa_vs_q"]
    assert math.isclose(s["q_new"], 3.25) and math.isclose(s["sarsa_new"], 2.98)


def test_reinforce_moves_toward_taken_action():
    r = build()["reinforce"]
    assert math.isclose(r["theta_new"][0], 0.1) and math.isclose(r["theta_new"][1], -0.1)
    assert math.isclose(r["pi_new"][0], 1 / (1 + math.exp(-0.2)))


def test_actor_critic_and_belief():
    b = build()
    assert math.isclose(b["actor_critic"]["delta"], 1.7)
    assert math.isclose(b["actor_critic"]["v_new"], 2.17)
    assert math.isclose(b["belief"]["b1"], 0.8)
    assert math.isclose(b["belief"]["b2"], 0.64 / 0.68)


def test_exercises():
    e = build()["exercises"]
    assert math.isclose(e["ex1_G"], 3.0) and math.isclose(e["ex2_delta"], 1.0)
    assert math.isclose(e["ex4_q_target"], 3.0) and math.isclose(e["ex4_sarsa_target"], 2.0)
    assert math.isclose(e["ex5_b1"], 0.7)
