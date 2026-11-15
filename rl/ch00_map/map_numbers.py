"""강화학습 0장(지도)의 손계산 숫자를 한곳에서 계산한다.

웹 챕터 tracks/rl/00-map.html의 모든 숫자는 results/ch00.json에서 옮긴다.
표준 라이브러리만 쓴다.

무엇을 재는지: 가치 갱신(동적 계획법, 몬테카를로, 시간차), SARSA와 Q-학습의 목표값 차이,
REINFORCE 한 걸음, 액터-크리틱 TD 오차, 믿음 갱신, 모형 기반 추정을 손으로 따라갈 수 있는 작은 값으로 계산한다.
입력: 없음(모든 숫자는 함수 안에 고정). 출력: results/ch00.json (소수 넷째 자리 반올림).
공통 기호: gamma = 할인율, alpha = 학습률, G = 할인 반환(누적 보상), delta = TD 오차.
"""
import json
import math
from pathlib import Path


def discounted_return(rewards, gamma):
    # 식: G = sum_k gamma^k * r_k  (k는 0부터, 첫 보상은 할인 없음)
    return sum((gamma ** k) * r for k, r in enumerate(rewards))


def chain_example():
    """A -> B -> 끝. A에서 B로 갈 때 보상 0, B에서 끝날 때 보상 1(확률 0.7) 또는 0(확률 0.3)."""
    gamma, alpha = 0.9, 0.5
    v = {"A": 0.5, "B": 0.6}
    p_win = 0.7
    # 식(참값, 벨만 기대 방정식): V(B) = 0.7*1 + 0.3*0, V(A) = 0 + gamma*V(B)
    true_b = p_win * 1 + (1 - p_win) * 0
    true_a = 0 + gamma * true_b
    # 동적 계획법: 모델로 한 번 훑기(동기식, 옛 값 사용)
    # 식(DP 한 번 훑기): V(s) <- E[r + gamma*V(s')]. 모델(확률 0.7)로 기댓값을 직접 계산.
    # 함정: 동기식이라 A는 갱신된 B(0.7)가 아니라 옛 값 v["B"]=0.6을 쓴다 (그래서 0.54, 참값 0.63 아님)
    dp = {"B": p_win * 1.0 + (1 - p_win) * 0.0, "A": 0 + gamma * v["B"]}
    # 몬테카를로: 에피소드 A,0,B,1,끝을 다 본 뒤
    # 식(MC 반환): G_A = 0 + gamma*1 (에피소드가 끝난 뒤에야 계산 가능)
    g_a = 0 + gamma * 1
    g_b = 1
    # 식(MC 갱신): V(s) <- V(s) + alpha*(G - V(s))
    mc = {"A": v["A"] + alpha * (g_a - v["A"]), "B": v["B"] + alpha * (g_b - v["B"])}
    # 시간차: 한 걸음마다 바로 고침(A를 먼저 고칠 때 B는 아직 옛 값)
    # 식(TD(0) 목표): r + gamma*V(s'). 다음 상태 값을 추정치 v["B"]로 대체(부트스트랩)
    td_target_a = 0 + gamma * v["B"]
    # 식(TD 갱신): V(s) <- V(s) + alpha*delta, delta = 목표 - V(s)
    td_a = v["A"] + alpha * (td_target_a - v["A"])
    # B는 종료 직전이라 V(끝)=0, 목표 = 1 + gamma*0
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
    # 학습률 alpha=0.1(위 chain의 0.5보다 작게 잡은 이 예시의 선택)
    q_sa, r, gamma, alpha = 3.0, 1.0, 0.9, 0.1
    q_next = {"left": 2.0, "right": 5.0}
    # SARSA는 실제로 고른 다음 행동 값, Q-학습은 최댓값을 쓴다는 차이를 보이려고 일부러 값이 낮은 쪽을 고른 상황을 가정
    taken_next = "left"  # ε-greedy가 탐험으로 왼쪽을 골랐다
    # 식(Q-학습, 오프폴리시): 목표 = r + gamma * max_a' Q(s',a')
    q_target = r + gamma * max(q_next.values())
    # 식(SARSA, 온폴리시): 목표 = r + gamma * Q(s',a'), a'는 실제로 선택된 행동
    s_target = r + gamma * q_next[taken_next]
    return {
        "q_sa": q_sa, "r": r, "gamma": gamma, "alpha": alpha, "q_next": q_next,
        "q_target": q_target, "q_new": q_sa + alpha * (q_target - q_sa),
        "sarsa_target": s_target, "sarsa_new": q_sa + alpha * (s_target - q_sa),
    }


def reinforce_step():
    # 정책 파라미터 theta(행동 2개), 초기엔 둘 다 0이라 pi = (0.5, 0.5)
    theta = [0.0, 0.0]
    z = [math.exp(t) for t in theta]
    # 식: pi_i = exp(theta_i) / sum_j exp(theta_j)  (소프트맥스 정책)
    pi = [x / sum(z) for x in z]
    # g = 관측한 반환 G, a = 실제로 고른 행동의 번호
    g, alpha, a = 2.0, 0.1, 0
    # 식: d log pi(a) / d theta_i = 1[i=a] - pi_i  (소프트맥스의 로그 기울기)
    grad = [(1 if i == a else 0) - pi[i] for i in range(2)]
    # 식(REINFORCE): theta <- theta + alpha * G * grad log pi(a)
    new = [theta[i] + alpha * g * grad[i] for i in range(2)]
    z2 = [math.exp(t) for t in new]
    pi2 = [x / sum(z2) for x in z2]
    return {"pi": pi, "G": g, "alpha": alpha, "grad": grad, "theta_new": new, "pi_new": pi2}


def actor_critic_step():
    # beta = 크리틱 학습률
    r, gamma, v_next, v_now, beta = 1.0, 0.9, 3.0, 2.0, 0.1
    # 식(TD 오차): delta = r + gamma*V(s') - V(s)
    delta = r + gamma * v_next - v_now
    # 식(크리틱 갱신): V(s) <- V(s) + beta*delta. 액터는 같은 delta를 G 대신 쓴다(여기선 크리틱만 계산)
    return {"delta": delta, "v_new": v_now + beta * delta}


def belief_update():
    # acc = 한 번의 관측이 맞을 확률(모형의 가정, 두 번 관측은 서로 독립)
    prior, acc = 0.5, 0.8
    # 식(베이즈): b' = acc*b / (acc*b + (1-acc)*(1-b))
    b1 = acc * prior / (acc * prior + (1 - acc) * (1 - prior))
    b2 = acc * b1 / (acc * b1 + (1 - acc) * (1 - b1))
    return {"prior": prior, "acc": acc, "b1": b1, "b2": b2}


def model_based():
    gamma = 0.9
    many = {"visits": 10, "wins": 7}
    few = {"visits": 3, "wins": 3}
    # 식: R_hat = wins / visits (표본 평균). 표본이 적을수록 불확실하다는 점이 요점
    r_many = many["wins"] / many["visits"]
    r_few = few["wins"] / few["visits"]
    # 식: V(A) = gamma * R_hat  (A에서 보상 0, 이후 B에서 R_hat를 받는 구조 가정)
    return {"many": many, "few": few, "R_hat_many": r_many, "V_A_many": gamma * r_many,
            "R_hat_few": r_few, "V_A_few": gamma * r_few}


def exercises():
    return {
        "ex1_G": discounted_return([2, 0, 4], 0.5),
        # 위 식들에 연습문제 값을 그대로 대입(ex2: delta = r + gamma*V(s') - V(s), r=0)
        "ex2_delta": 0 + 0.9 * 10 - 8,
        "ex4_q_target": 0 + 0.5 * max(4, 6), "ex4_sarsa_target": 0 + 0.5 * 4,
        # ex5: 정확도 0.7, 사전 0.5로 같은 베이즈 식
        "ex5_b1": 0.7 * 0.5 / (0.7 * 0.5 + 0.3 * 0.5),
    }


def build():
    return {
        # 본문 예시: 보상 1,1,10과 gamma=0.9의 반환 (1 + 0.9 + 0.81*10 = 10.0)
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
    # 표시용 반올림. 테스트는 build()의 비반올림 값을 isclose로 비교
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
