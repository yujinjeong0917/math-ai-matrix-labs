"""10장: 기록(trace) 20개를 결과만 채점할 때와 과정까지 채점할 때 비교한다. 전부 가상 기록이에요."""
import math

APPROVAL_THRESHOLD = 50   # 가상 문턱: 이 금액 이상이면 사람 승인이 먼저 있어야 해요

# 문의 20건: (번호, 주문번호, 정답 환불 금액)
TICKETS = [(i + 1, 100 + i + 1, a) for i, a in enumerate(
    [30, 20, 50, 40, 25, 60, 15, 35, 45, 10, 70, 55, 20, 80, 35, 25, 65, 40, 30, 90])]

# 기록 종류(v1: 고치기 전 에이전트). 문제는 문의 번호 기준으로 하나씩 심었고, 나머지 문의는 정상(ok)이에요.
# ok_same_key_twice 같은 키로 두 번 호출(서버가 한 번만 처리) / skip_approval 승인 건너뜀(50 이상 금액만 해당) /
# no_lookup 조회 없이 환불 / dup_new_key 새 키로 두 번 환불 / wrong_amount 금액을 5 더 보냄
V1_PROBLEMS = {1: 'wrong_amount', 2: 'no_lookup', 3: 'skip_approval', 5: 'ok_same_key_twice',
               6: 'skip_approval', 8: 'no_lookup', 9: 'dup_new_key', 11: 'skip_approval'}


def make_trace(ticket, kind):
    n, order, amount = ticket
    ev = [{'tool': 'get_order', 'order': order}]
    needs_approval = amount >= APPROVAL_THRESHOLD
    if kind == 'no_lookup':
        ev = []
    if needs_approval and kind != 'skip_approval':
        ev.append({'tool': 'approve', 'order': order, 'ok': True})
    paid = amount + 5 if kind == 'wrong_amount' else amount
    ev.append({'tool': 'refund', 'order': order, 'amount': paid, 'key': f'refund-{order}'})
    if kind == 'ok_same_key_twice':
        ev.append({'tool': 'refund', 'order': order, 'amount': paid, 'key': f'refund-{order}'})
    if kind == 'dup_new_key':
        ev.append({'tool': 'refund', 'order': order, 'amount': paid, 'key': f'refund-{order}-retry'})
    ev.append({'tool': 'notify', 'order': order})
    return {'ticket': n, 'order': order, 'expected': amount, 'kind': kind, 'events': ev}


def build_traces(kinds_by_ticket):
    return [make_trace(t, kinds_by_ticket[t[0]]) for t in TICKETS]


def v1_kinds():
    return {t[0]: V1_PROBLEMS.get(t[0], 'ok') for t in TICKETS}


def v2_kinds():
    """규칙을 고친 뒤(v2): 승인 건너뜀 3건을 고쳤지만, 정상이던 4번 문의가 금액을 5 더 보내요."""
    kinds = dict(v1_kinds())
    for n, k in kinds.items():
        if k == 'skip_approval':
            kinds[n] = 'ok'
    kinds[4] = 'wrong_amount'
    return kinds


def replay_payouts(trace):
    """환불 서버를 흉내 내서 실제로 나간 돈을 센다. 같은 키는 한 번만 나가요."""
    seen, paid = set(), 0
    for e in trace['events']:
        if e['tool'] == 'refund' and e['key'] not in seen:
            seen.add(e['key'])
            paid += e['amount']
    return paid


def grade_outcome(trace):
    return replay_payouts(trace) == trace['expected']


def process_failures(trace):
    fails, tools_so_far, approved = [], [], False
    keys = {}
    for e in trace['events']:
        if e['tool'] == 'approve' and e.get('ok'):
            approved = True
        if e['tool'] == 'refund':
            if 'get_order' not in tools_so_far:
                fails.append('조회 없이 환불')
            if e['amount'] >= APPROVAL_THRESHOLD and not approved:
                fails.append('승인 없이 환불')
            keys.setdefault(e['order'], set()).add(e['key'])
        tools_so_far.append(e['tool'])
    if any(len(k) > 1 for k in keys.values()):
        fails.append('같은 주문에 키가 다른 환불 두 번')
    return fails


def grade_process(trace):
    return not process_failures(trace)


def wilson(passed, n, z=1.96):
    """통과율의 95% 신뢰구간(윌슨 방식). 표준 라이브러리만 써요."""
    p = passed / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def summarize(traces):
    n = len(traces)
    out_ok = [grade_outcome(t) for t in traces]
    proc_ok = [grade_process(t) for t in traces]
    both = [a and b for a, b in zip(out_ok, proc_ok)]
    return {
        'n': n,
        'outcome_pass': sum(out_ok), 'process_pass': sum(proc_ok), 'both_pass': sum(both),
        'outcome_only_pass_but_process_fail': sum(a and not b for a, b in zip(out_ok, proc_ok)),
        'ci_outcome': wilson(sum(out_ok), n), 'ci_both': wilson(sum(both), n),
        'passed_tickets': [t['ticket'] for t, b in zip(traces, both) if b],
        'failure_reasons': {t['ticket']: process_failures(t) for t in traces if process_failures(t)},
    }


def regression(before, after):
    """고치기 전에 통과했는데 고친 뒤 실패한 문의 번호."""
    return [b['ticket'] for b, a in zip(before, after)
            if grade_outcome(b) and grade_process(b) and not (grade_outcome(a) and grade_process(a))]


def report():
    v1 = build_traces(v1_kinds())
    v2 = build_traces(v2_kinds())
    s1, s2 = summarize(v1), summarize(v2)
    return {'v1': s1, 'v2': s2, 'regressions_v1_to_v2': regression(v1, v2),
            'ci_v2_both': s2['ci_both']}


if __name__ == '__main__':
    import json
    print(json.dumps(report(), ensure_ascii=False, indent=1))
