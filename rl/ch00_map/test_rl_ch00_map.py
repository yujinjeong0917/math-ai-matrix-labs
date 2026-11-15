import math
from map_numbers import build, discounted_return


# 1 + 0.9*1 + 0.81*10 = 10.0 (할인 반환 손계산)
def test_return_example():
    assert math.isclose(discounted_return([1, 1, 10], 0.9), 10.0)


def test_chain_numbers():
    c = build()["chain"]
    # 참값: V(B)=0.7, V(A)=0.9*0.7=0.63
    assert math.isclose(c["true"]["A"], 0.63) and math.isclose(c["true"]["B"], 0.7)
    # MC: G_A=0.9, G_B=1 에서 V += 0.5*(G-V): 0.5->0.7, 0.6->0.8
    assert math.isclose(c["mc"]["A"], 0.7) and math.isclose(c["mc"]["B"], 0.8)
    # TD: A 목표 0.9*0.6=0.54, 0.5+0.5*0.04=0.52 / B 목표 1, 0.6+0.5*0.4=0.8
    assert math.isclose(c["td"]["A"], 0.52) and math.isclose(c["td"]["B"], 0.8)
    # DP: 동기식이라 A는 옛 V(B)=0.6 사용, 0.9*0.6=0.54 (참값 0.63과 다름)
    assert math.isclose(c["dp"]["A"], 0.54) and math.isclose(c["dp"]["B"], 0.7)


def test_sarsa_vs_q():
    s = build()["sarsa_vs_q"]
    # Q: 목표 1+0.9*5=5.5, 3+0.1*2.5=3.25 / SARSA: 목표 1+0.9*2=2.8, 3+0.1*(-0.2)=2.98
    assert math.isclose(s["q_new"], 3.25) and math.isclose(s["sarsa_new"], 2.98)


def test_reinforce_moves_toward_taken_action():
    r = build()["reinforce"]
    # 기울기 (1-0.5, -0.5)=(0.5,-0.5), 변화 alpha*G*grad = 0.1*2*0.5 = 0.1
    assert math.isclose(r["theta_new"][0], 0.1) and math.isclose(r["theta_new"][1], -0.1)
    assert math.isclose(r["pi_new"][0], 1 / (1 + math.exp(-0.2)))


def test_actor_critic_and_belief():
    b = build()
    # delta = 1 + 0.9*3 - 2 = 1.7, v_new = 2 + 0.1*1.7 = 2.17
    assert math.isclose(b["actor_critic"]["delta"], 1.7)
    assert math.isclose(b["actor_critic"]["v_new"], 2.17)
    # b1 = 0.8*0.5/(0.8*0.5+0.2*0.5) = 0.8, b2 = 0.64/(0.64+0.04) = 0.64/0.68
    assert math.isclose(b["belief"]["b1"], 0.8)
    assert math.isclose(b["belief"]["b2"], 0.64 / 0.68)


def test_exercises():
    e = build()["exercises"]
    # ex1: 2 + 0 + 0.25*4 = 3, ex2: 0 + 9 - 8 = 1
    assert math.isclose(e["ex1_G"], 3.0) and math.isclose(e["ex2_delta"], 1.0)
    # ex4: Q 목표 0.5*max(4,6)=3, SARSA 목표 0.5*4=2
    assert math.isclose(e["ex4_q_target"], 3.0) and math.isclose(e["ex4_sarsa_target"], 2.0)
    assert math.isclose(e["ex5_b1"], 0.7)
