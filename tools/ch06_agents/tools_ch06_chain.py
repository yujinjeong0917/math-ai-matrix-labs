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


def chain(p, n):
    """n단계가 모두 옳을 확률 p^n (단계끼리 서로 영향이 없다고 본다)."""
    return F(p) ** n


def first_failure_at(p, i):
    """확인 없이 돌릴 때 처음 틀리는 단계가 i번째일 확률 p^(i-1)(1-p)."""
    return F(p) ** (i - 1) * (1 - F(p))


def max_steps(p, target):
    """p^n >= target 을 지키는 가장 큰 n."""
    n = 0
    while chain(p, n + 1) >= target:
        n += 1
    return n


def step_outcomes(p, d, k):
    """확인 지점이 붙은 단계 하나의 결과 세 갈래와 평균 시도 횟수.

    한 번 시도하면: 옳음 p / 틀렸고 확인에 잡힘 c = (1-p)d / 틀렸는데 놓침 m = (1-p)(1-d).
    잡히면 다시 시도하고, k번 다시 해도 또 잡히면 거기서 멈추고 사람에게 넘긴다.
    """
    p, d = F(p), F(d)
    c = (1 - p) * d
    m = (1 - p) * (1 - d)
    g = sum(c ** j for j in range(k + 1))          # 1 + c + ... + c^k
    return {"correct": p * g, "missed": m * g, "stopped": c ** (k + 1), "attempts": g}


def run_outcomes(per_step, human=None, gate=None):
    """단계 결과 목록으로 한 번 돌린 결과(끝까지 옳음 / 틀린 채 끝까지 감 / 멈춤)를 낸다.

    per_step: 단계마다 step_outcomes 결과. 놓친 틀림은 멈추지 않고 다음 단계로 간다.
    human, gate: gate번째 단계(0부터) 바로 앞에서 사람이 지금까지의 결과를 보고, 틀렸으면 human의 확률로 알아채 멈춘다.
    """
    ok = F(1)                  # 여기까지 모두 옳게 온 확률
    wr = F(0)                  # 여기까지 멈추지 않았지만 어딘가 틀린 채 온 확률
    stop_at, caught, attempts, approvals = [], F(0), F(0), F(0)
    for i, s in enumerate(per_step):
        if human is not None and i == gate:
            approvals = ok + wr
            caught = wr * F(human)
            wr -= caught
        go = ok + wr
        attempts += go * s["attempts"]
        stop_at.append(go * s["stopped"])
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
    ag_n, ag_o = (p * q) ** n, (p * q) ** (n + 1)
    even = (wf_n - ag_n) / (wf_n - ag_n + ag_o - wf_o)
    return {"workflow_normal": wf_n, "workflow_odd": wf_o, "agent_normal": ag_n, "agent_odd": ag_o,
            "workflow_mix": (1 - f) * wf_n + f * wf_o, "agent_mix": (1 - f) * ag_n + f * ag_o,
            "break_even_odd_share": even}
