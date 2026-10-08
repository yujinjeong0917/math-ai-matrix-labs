"""강화학습 0장(지도)의 손계산 숫자를 한곳에서 계산한다.

웹 챕터 tracks/rl/00-map.html의 모든 숫자는 results/ch00.json에서 옮긴다.
표준 라이브러리만 쓴다.
"""
import json
import math
from pathlib import Path


def discounted_return(rewards, gamma):
    return sum((gamma ** k) * r for k, r in enumerate(rewards))


def chain_example():
    """A -> B -> 끝. A에서 B로 갈 때 보상 0, B에서 끝날 때 보상 1(확률 0.7) 또는 0(확률 0.3)."""
    gamma, alpha = 0.9, 0.5
    v = {"A": 0.5, "B": 0.6}
    p_win = 0.7
    true_b = p_win * 1 + (1 - p_win) * 0
    true_a = 0 + gamma * true_b
    # 동적 계획법: 모델로 한 번 훑기(동기식, 옛 값 사용)
    dp = {"B": p_win * 1.0 + (1 - p_win) * 0.0, "A": 0 + gamma * v["B"]}
    # 몬테카를로: 에피소드 A,0,B,1,끝을 다 본 뒤
    g_a = 0 + gamma * 1
    g_b = 1
    mc = {"A": v["A"] + alpha * (g_a - v["A"]), "B": v["B"] + alpha * (g_b - v["B"])}
    # 시간차: 한 걸음마다 바로 고침(A를 먼저 고칠 때 B는 아직 옛 값)
    td_target_a = 0 + gamma * v["B"]
    td_a = v["A"] + alpha * (td_target_a - v["A"])
    td_target_b = 1 + gamma * 0
    td_b = v["B"] + alpha * (td_target_b - v["B"])
    return {
        "gamma": gamma, "alpha": alpha, "v0": v, "p_win": p_win,
        "true": {"A": true_a, "B": true_b},
        "dp": dp,
        "mc": {"G_A": g_a, "G_B": g_b, **{k: mc[k] for k in mc}},
        "td": {"target_A": td_target_a, "delta_A": td_target_a - v["A"], "A": td_a,
               "target_B": td_target_b, "B": td_b},
    }


def sarsa_vs_q():
    q_sa, r, gamma, alpha = 3.0, 1.0, 0.9, 0.1
    q_next = {"left": 2.0, "right": 5.0}
    taken_next = "left"  # ε-greedy가 탐험으로 왼쪽을 골랐다
    q_target = r + gamma * max(q_next.values())
    s_target = r + gamma * q_next[taken_next]
    return {
        "q_sa": q_sa, "r": r, "gamma": gamma, "alpha": alpha, "q_next": q_next,
        "q_target": q_target, "q_new": q_sa + alpha * (q_target - q_sa),
        "sarsa_target": s_target, "sarsa_new": q_sa + alpha * (s_target - q_sa),
    }


def reinforce_step():
    theta = [0.0, 0.0]
    z = [math.exp(t) for t in theta]
    pi = [x / sum(z) for x in z]
    g, alpha, a = 2.0, 0.1, 0
    grad = [(1 if i == a else 0) - pi[i] for i in range(2)]
    new = [theta[i] + alpha * g * grad[i] for i in range(2)]
    z2 = [math.exp(t) for t in new]
    pi2 = [x / sum(z2) for x in z2]
    return {"pi": pi, "G": g, "alpha": alpha, "grad": grad, "theta_new": new, "pi_new": pi2}


def actor_critic_step():
    r, gamma, v_next, v_now, beta = 1.0, 0.9, 3.0, 2.0, 0.1
    delta = r + gamma * v_next - v_now
    return {"delta": delta, "v_new": v_now + beta * delta}


def belief_update():
    prior, acc = 0.5, 0.8
    b1 = acc * prior / (acc * prior + (1 - acc) * (1 - prior))
    b2 = acc * b1 / (acc * b1 + (1 - acc) * (1 - b1))
    return {"prior": prior, "acc": acc, "b1": b1, "b2": b2}


def model_based():
    gamma = 0.9
    many = {"visits": 10, "wins": 7}
    few = {"visits": 3, "wins": 3}
    r_many = many["wins"] / many["visits"]
    r_few = few["wins"] / few["visits"]
    return {"many": many, "few": few, "R_hat_many": r_many, "V_A_many": gamma * r_many,
            "R_hat_few": r_few, "V_A_few": gamma * r_few}


def exercises():
    return {
        "ex1_G": discounted_return([2, 0, 4], 0.5),
        "ex2_delta": 0 + 0.9 * 10 - 8,
        "ex4_q_target": 0 + 0.5 * max(4, 6), "ex4_sarsa_target": 0 + 0.5 * 4,
        "ex5_b1": 0.7 * 0.5 / (0.7 * 0.5 + 0.3 * 0.5),
    }


def build():
    return {
        "return_example": {"rewards": [1, 1, 10], "gamma": 0.9, "G": discounted_return([1, 1, 10], 0.9)},
        "chain": chain_example(),
        "sarsa_vs_q": sarsa_vs_q(),
        "reinforce": reinforce_step(),
        "actor_critic": actor_critic_step(),
        "belief": belief_update(),
        "model_based": model_based(),
        "exercises": exercises(),
    }


def rounded(x, nd=4):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: rounded(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v, nd) for v in x]
    return x


if __name__ == "__main__":
    out = Path(__file__).parent / "results" / "ch00.json"
    out.write_text(json.dumps(rounded(build()), ensure_ascii=False, indent=1))
    print(out.read_text())
