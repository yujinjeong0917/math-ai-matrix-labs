"""AI 도구 실전 6장: 단계가 이어질 때의 성공 확률을 닫힌 식(분수)으로 계산한다.

모든 확률은 fractions.Fraction이라 반올림 오차가 없다.
기호
  p  단계 하나를 옳게 해낼 확률
  d  확인 지점(검증 테스트)이 틀린 결과를 잡아낼 확률
  k  잡혔을 때 다시 해 보는 최대 횟수(시도는 최대 k + 1번)
  h  사람이 승인 전에 틀린 보고서를 알아챌 확률
  q  에이전트가 다음 행동을 옳게 고를 확률
"""

import json
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
RUN_FILE = HERE / "data" / "weekly_run.json"


def load_run(path=RUN_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assumptions(run=None):
    a = (run or load_run())["assume"]
    return {"p": F(a["p"]), "q": F(a["q"]), "d": F(a["d"]), "k": int(a["k"]), "h": F(a["h"]),
            "f": F(a["odd_week_share"])}


# 식: P(n단계 모두 옳음) = p^n. 단계가 서로 독립이라는 가정(실제 단계는 앞 단계 오류에 영향받을 수 있다).
def chain(p, n):
    """n단계가 모두 옳을 확률 p^n (단계끼리 서로 영향이 없다고 본다)."""
    return F(p) ** n


# 기하분포: 앞 i-1단계는 옳고(p^(i-1)) i번째에서 처음 틀린다((1-p)).
def first_failure_at(p, i):
    """확인 없이 돌릴 때 처음 틀리는 단계가 i번째일 확률 p^(i-1)(1-p)."""
    return F(p) ** (i - 1) * (1 - F(p))


def max_steps(p, target):
    """p^n >= target 을 지키는 가장 큰 n."""
    n = 0
    # 불변식: 루프에 들어올 때 chain(p, n) >= target. 단조 감소(p<1)라 처음 어긋나는 곳에서 멈추면 최대 n. 시간복잡도 O(n).
    while chain(p, n + 1) >= target:
        n += 1
    return n


def step_outcomes(p, d, k):
    """확인 지점이 붙은 단계 하나의 결과 세 갈래와 평균 시도 횟수.

    한 번 시도하면: 옳음 p / 틀렸고 확인에 잡힘 c = (1-p)d / 틀렸는데 놓침 m = (1-p)(1-d).
    잡히면 다시 시도하고, k번 다시 해도 또 잡히면 거기서 멈추고 사람에게 넘긴다.
    """
    p, d = F(p), F(d)
    # 한 번 시도의 세 갈래 확률: 옳음 p, 잡힘 c, 놓침 m. 셋의 합 = p + (1-p) = 1.
    c = (1 - p) * d
    m = (1 - p) * (1 - d)
    # 식: 다시 하기를 k번까지 허용하면 '잡혀서 다시 한' 경우만큼 시도가 이어진다. g = 1 + c + ... + c^k는 등비합이고, 옳음/놓침 확률에 곱해진다. 평균 시도 횟수도 g.
    g = sum(c ** j for j in range(k + 1))          # 1 + c + ... + c^k
    # 멈춤 확률 c^(k+1): k+1번 연속 잡히는 경우. correct + missed + stopped = 1이 테스트에서 확인된다.
    return {"correct": p * g, "missed": m * g, "stopped": c ** (k + 1), "attempts": g}


def run_outcomes(per_step, human=None, gate=None):
    """단계 결과 목록으로 한 번 돌린 결과(끝까지 옳음 / 틀린 채 끝까지 감 / 멈춤)를 낸다.

    per_step: 단계마다 step_outcomes 결과. 놓친 틀림은 멈추지 않고 다음 단계로 간다.
    human, gate: gate번째 단계(0부터) 바로 앞에서 사람이 지금까지의 결과를 보고, 틀렸으면 human의 확률로 알아채 멈춘다.
    """
    # 불변식: 각 단계 앞에서 ok(전부 옳게 옴) + wr(틀렸지만 못 알아챔) + 이미 멈춘 확률 = 1. 단계 수 n에 대해 O(n).
    ok = F(1)                  # 여기까지 모두 옳게 온 확률
    wr = F(0)                  # 여기까지 멈추지 않았지만 어딘가 틀린 채 온 확률
    stop_at, caught, attempts, approvals = [], F(0), F(0), F(0)
    for i, s in enumerate(per_step):
        if human is not None and i == gate:
            # 승인 단계에서는 아직 멈추지 않은 모든 경우가 승인 요청을 받는다. 그중 틀린 경우(wr)만 h의 확률로 걸러진다.
            approvals = ok + wr
            caught = wr * F(human)
            wr -= caught
        go = ok + wr
        attempts += go * s["attempts"]
        stop_at.append(go * s["stopped"])
        # 갱신: 계속 옳음 = ok*correct. 틀린 채 감 = ok*missed(이번에 처음 놓침) + wr*(correct+missed)(이미 틀린 상태에서 이어감). 멈춘 몫은 stop_at에 따로 쌓았다.
        ok, wr = ok * s["correct"], ok * s["missed"] + wr * (s["correct"] + s["missed"])
    return {"all_correct": ok, "delivered_wrong": wr, "stopped_by_check": sum(stop_at, F(0)),
            "stopped_by_human": caught, "stop_at": stop_at, "expected_attempts": attempts,
            "approvals": approvals}


def uniform(n, p, d=0, k=0):
    return [step_outcomes(p, d, k) for _ in range(n)]


def per_step_retry_perfect(p, k):
    """확인이 늘 잡을 때(d = 1) 단계 하나가 옳을 확률 1 - (1-p)^(k+1)."""
    return 1 - (1 - F(p)) ** (k + 1)


def agent_vs_workflow(p, q, n, f):
    """보통 주(n단계)와 이상한 주(문의가 3쪽이라 읽기가 하나 더 필요)에서 끝까지 옳을 확률.

    워크플로: 정해진 n단계를 그대로 돈다. 이상한 주에는 3쪽을 읽지 않으니 끝까지 옳을 수 없다(0).
    에이전트: 매 단계 다음 행동을 고르고(q) 해낸다(p). 이상한 주에는 n + 1단계가 된다.
    f: 이상한 주의 비율. 반환값의 even은 두 방식의 평균 성공률이 같아지는 f.
    """
    p, q, f = F(p), F(q), F(f)
    wf_n, wf_o = p ** n, F(0)
    # 에이전트는 단계마다 고르기(q)와 해내기(p)가 곱해져 (pq)^n. 이상한 주는 읽기 단계가 하나 늘어 n+1.
    ag_n, ag_o = (p * q) ** n, (p * q) ** (n + 1)
    # 두 평균 (1-f)*wf_n + f*wf_o = (1-f)*ag_n + f*ag_o를 f에 대해 풀면 f = (wf_n-ag_n)/(wf_n-ag_n+ag_o-wf_o). 분모가 0이면 나눗셈 오류가 나므로 입력 가정에 주의.
    even = (wf_n - ag_n) / (wf_n - ag_n + ag_o - wf_o)
    return {"workflow_normal": wf_n, "workflow_odd": wf_o, "agent_normal": ag_n, "agent_odd": ag_o,
            "workflow_mix": (1 - f) * wf_n + f * wf_o, "agent_mix": (1 - f) * ag_n + f * ag_o,
            "break_even_odd_share": even}
