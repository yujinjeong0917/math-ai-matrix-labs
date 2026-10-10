"""AI 도구 실전 6장: 주간 보고서 반복을 워크플로와 에이전트로 돌려 보는 시뮬레이션(고정 시드).

모델도 Notion·Figma도 부르지 않는다. 단계가 옳게 되는지는 난수(random.Random(시드))로 정한다.

- 워크플로: data/weekly_run.json의 22단계를 적힌 순서대로 돈다. 다음 단계를 코드가 정한다.
- 에이전트: 3장 tools_ch03_loop.py의 도구 호출 고리 모양(stop_reason "tool_use" -> 실행 -> tool_result ->
  다시 요청, "end_turn"이면 끝)을 그대로 옮겼다. 다음 행동은 가짜 모델(ScriptedAgent)이 지금까지 본 결과로 고른다.
  예를 들어 Notion이 next_cursor를 주면 한 쪽을 더 읽는다. 다만 매번 q의 확률로만 옳게 고르고,
  잘못 고르면 그 일을 건너뛴다(틀린 채 넘어감).
- 확인 지점: 단계마다 검증 테스트가 틀린 결과를 d의 확률로 잡는다. 잡히면 최대 k번 다시 하고,
  그래도 안 되면 그 단계에서 멈추고 사람에게 넘긴다. 워크플로에서는 "Notion을 다 읽었나(next_cursor가 비었나)"
  확인도 함께 해서, 3쪽짜리 주에는 2쪽을 읽은 뒤 멈춘다.
- 사람 승인: 되돌리기 어려운 단계(undo = hard, 팀 채널 알림) 바로 앞에서 사람이 보고서를 본다.
  틀린 보고서를 h의 확률로 알아채고 멈춘다.
"""

import math

import tools_ch06_chain as C

SEED = 6


class Week:
    """한 주의 상황. rows가 100을 넘을 때마다 Notion 읽기가 한 쪽씩 늘어난다."""

    def __init__(self, rows, per_page=100):
        self.rows = rows
        self.per_page = per_page
        self.pages = math.ceil(rows / per_page)


def attempt(rng, p_ok, d, k, checks):
    """단계 하나를 시도한다. ('ok' | 'missed' | 'stopped', 시도 횟수)."""
    tries = 0
    while True:
        tries += 1
        if rng.random() < p_ok:
            return "ok", tries
        if checks and rng.random() < d:
            if tries <= k:
                continue
            return "stopped", tries
        return "missed", tries


def gate(log, wrong, h, rng):
    """되돌리기 어려운 단계 바로 앞의 사람 승인. 멈추면 True."""
    log["approvals"] += 1
    if wrong and rng.random() < h:
        log.update(outcome="stopped", stop_step="share", cause="사람 승인에서 틀린 보고서를 알아챔")
        return True
    return False


def run_workflow(steps, week, rng, p, d=0, k=0, checks=False, human=False, h=0):
    """정해진 순서대로 돈다. 반환: 결과(correct/wrong/stopped), 멈춘 단계, 까닭, 시도 수, 승인 수."""
    log = {"mode": "workflow", "attempts": 0, "approvals": 0, "stop_step": None, "cause": None}
    wrong = False
    for s in steps:
        if human and s["undo"] == "hard" and gate(log, wrong, h, rng):
            return log
        res, tries = attempt(rng, p, d, k, checks)
        log["attempts"] += tries
        if res == "stopped":
            log.update(outcome="stopped", stop_step=s["id"], cause="확인 지점에서 다시 해도 계속 틀림")
            return log
        wrong |= res == "missed"
        if s["id"] == "notion_2" and week.pages > 2:
            if checks:   # next_cursor가 남았는지 보는 확인. 결정적이라 늘 잡는다
                log.update(outcome="stopped", stop_step="notion_2", cause="Notion에 읽지 않은 쪽이 남음")
                return log
            wrong = True   # 3쪽을 읽지 않고 넘어감: 문의 일부가 빠진 보고서
    log["outcome"] = "wrong" if wrong else "correct"
    return log


class ScriptedAgent:
    """지금까지 본 도구 결과로 다음 행동을 고르는 가짜 모델. q의 확률로만 옳게 고른다."""

    def __init__(self, steps, rng, q):
        self.base = list(steps)
        self.rng = rng
        self.q = q

    def next_needed(self, done, cursor_left):
        reads = [s for s in done if s.startswith("notion_")]
        if cursor_left:                                   # Notion이 다음 쪽이 있다고 했으면 더 읽는다
            return {"id": f"notion_{len(reads) + 1}", "tool": "query_inquiries"}
        for s in self.base:
            if s["id"].startswith("notion_"):
                continue
            if s["id"] not in done:
                return {"id": s["id"], "tool": s["tool"], "undo": s["undo"]}
        return None

    def create(self, done, cursor_left):
        nxt = self.next_needed(done, cursor_left)
        if nxt is None:
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "보고서를 다 만들었어요."}]}
        right = self.rng.random() < self.q
        return {"stop_reason": "tool_use", "right_choice": right,
                "undo": nxt.get("undo", "easy"),
                "content": [{"type": "tool_use", "id": f"toolu_{nxt['id']}", "name": nxt["tool"] or "think",
                             "input": {"step": nxt["id"]}}]}


def run_agent(steps, week, rng, p, q, d=0, k=0, checks=False, human=False, h=0, max_turns=100):
    """3장의 도구 호출 고리 모양으로 돈다. 다음 행동은 모델(가짜)이 고른다."""
    model = ScriptedAgent(steps, rng, q)
    log = {"mode": "agent", "attempts": 0, "approvals": 0, "stop_step": None, "cause": None, "turns": 0}
    done, wrong = [], False
    read_pages = 0
    while True:
        cursor_left = read_pages < week.pages      # 처음(0쪽)이거나 next_cursor가 남았으면 True
        resp = model.create(done, cursor_left)
        log["turns"] += 1
        if resp["stop_reason"] == "end_turn" or log["turns"] > max_turns:
            break
        step_id = resp["content"][0]["input"]["step"]
        if human and resp["undo"] == "hard" and gate(log, wrong, h, rng):
            return log
        # 한 번 시도 = 옳게 고르고(q) 옳게 해내기(p). 확인 지점은 둘 중 무엇이 틀렸든 같은 확률로 잡는다.
        tries = 0
        while True:
            tries += 1
            ok = resp["right_choice"] and rng.random() < p
            if ok:
                res = "ok"
                break
            if checks and rng.random() < d:
                if tries <= k:
                    resp = {"right_choice": rng.random() < q}
                    continue
                res = "stopped"
                break
            res = "missed"
            break
        log["attempts"] += tries
        if res == "stopped":
            log.update(outcome="stopped", stop_step=step_id, cause="확인 지점에서 다시 해도 계속 틀림")
            return log
        wrong |= res == "missed"
        done.append(step_id)
        if step_id.startswith("notion_"):
            read_pages += 1        # 건너뛰었든 아니든 결과에 next_cursor가 보이면 다음 쪽으로 간다
    log["outcome"] = "wrong" if wrong else "correct"
    return log


def simulate(runner, n_weeks, seed=SEED, odd_share=None, rows_normal=120, rows_odd=230, **kw):
    """여러 주를 돌려 결과를 센다. odd_share가 None이면 모두 보통 주, 1이면 모두 이상한 주."""
    import random
    rng = random.Random(seed)
    run = C.load_run()
    steps = run["steps"]
    tally = {"correct": 0, "wrong": 0, "stopped": 0}
    stop_steps, attempts, approvals = {}, 0, 0
    for _ in range(n_weeks):
        odd = odd_share is not None and rng.random() < odd_share
        week = Week(rows_odd if odd else rows_normal)
        log = runner(steps, week, rng, **kw)
        tally[log["outcome"]] += 1
        attempts += log["attempts"]
        approvals += log["approvals"]
        if log["outcome"] == "stopped":
            stop_steps[log["stop_step"]] = stop_steps.get(log["stop_step"], 0) + 1
    return {"weeks": n_weeks, **tally, "attempts": attempts, "approvals": approvals, "stop_steps": stop_steps}
