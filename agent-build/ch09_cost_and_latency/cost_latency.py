"""9장: 문의 한 건의 호출 수, 입력 누적, 대기 시간을 센다. 전부 가상 값이고 단가는 입력 변수예요."""

# ---- 입력값(가상). 실제 모델·서버의 값이 아니에요 ----
PARAMS = {
    'system_tokens': 800,      # 시스템 프롬프트 + 도구 정의(매 호출마다 다시 보내요)
    'user_tokens': 60,         # 고객 문의 한 개
    'model_base_s': 0.5,       # 모델 호출 한 번의 고정 대기(초)
    'model_s_per_in_token': 0.0004,   # 입력 토큰 1개당 대기(초)
    'model_s_per_out_token': 0.02,    # 출력 토큰 1개당 대기(초)
    'approval_wait_s': 0.0,    # 사람 승인 대기(초). 5장의 값을 더하는 자리
}
TOOLS = {  # 도구: (결과 토큰 수, 응답 대기 초)
    'get_order': (150, 0.3),
    'get_customer': (120, 0.3),
    'get_policy': (400, 0.2),
    'refund': (60, 1.5),
    'refund_error': (40, 0.5),
}
STEP_OUT = 40   # 도구를 부르는 호출의 출력 토큰
REPLY_OUT = 120  # 마지막 안내 문장의 출력 토큰


def run_flow(calls, params=PARAMS):
    """calls: [{'out': 출력 토큰, 'tools': [도구 이름...], 'parallel': 참/거짓}, ...]
    돌려주는 값: 호출 수, 입력 토큰 합, 출력 토큰 합, 대기 시간(초)."""
    history = params['user_tokens']
    total_in = total_out = 0
    seconds = params['approval_wait_s']
    for call in calls:
        input_tokens = params['system_tokens'] + history      # 이력을 통째로 다시 보내요
        total_in += input_tokens
        total_out += call['out']
        seconds += (params['model_base_s'] + input_tokens * params['model_s_per_in_token']
                    + call['out'] * params['model_s_per_out_token'])
        history += call['out']
        waits = [TOOLS[t][1] for t in call['tools']]
        if waits:
            seconds += max(waits) if call.get('parallel') else sum(waits)
        history += sum(TOOLS[t][0] for t in call['tools'])
    return {'calls': len(calls), 'input_tokens': total_in, 'output_tokens': total_out,
            'seconds': round(seconds, 3)}


def cost_units(result, price_in, price_out):
    """비용 = 입력 토큰 합 x 입력 단가 + 출력 토큰 합 x 출력 단가. 단가는 부르는 쪽이 정해요."""
    return result['input_tokens'] * price_in + result['output_tokens'] * price_out


def serial_flow():
    return [{'out': STEP_OUT, 'tools': ['get_order']}, {'out': STEP_OUT, 'tools': ['get_customer']},
            {'out': STEP_OUT, 'tools': ['get_policy']}, {'out': STEP_OUT, 'tools': ['refund']},
            {'out': REPLY_OUT, 'tools': []}]


def parallel_flow():
    return [{'out': 3 * STEP_OUT, 'tools': ['get_order', 'get_customer', 'get_policy'], 'parallel': True},
            {'out': STEP_OUT, 'tools': ['refund']}, {'out': REPLY_OUT, 'tools': []}]


def flow_with_refund_failures(failures, cap=None):
    """환불 도구가 failures번 오류를 돌려준 뒤 성공해요. failures=None이면 계속 오류예요.
    cap은 모델 호출 단계 상한(max steps)이에요.
    돌려주는 값: (호출 목록, 환불이 나갔나, 고객 안내까지 끝났나)."""
    calls = [{'out': STEP_OUT, 'tools': ['get_order']}, {'out': STEP_OUT, 'tools': ['get_customer']},
             {'out': STEP_OUT, 'tools': ['get_policy']}]
    n = 0
    while True:
        if cap is not None and len(calls) >= cap:
            return calls, False, False
        if failures is not None and n >= failures:
            calls.append({'out': STEP_OUT, 'tools': ['refund']})
            break
        calls.append({'out': STEP_OUT, 'tools': ['refund_error']})
        n += 1
    if cap is not None and len(calls) >= cap:
        return calls, True, False   # 환불은 나갔는데 안내할 단계가 없어요
    calls.append({'out': REPLY_OUT, 'tools': []})
    return calls, True, True


# 가상 문의 20건: 환불 도구 오류 횟수. None은 끝까지 오류.
FAILURES = [0] * 12 + [1] * 4 + [2] * 2 + [5] + [None]
CAPS = [4, 5, 6, 8, 10, 20]


def stopped_count(cap):
    """상한 cap에서 안내까지 가지 못하고 멈춘 문의 수와, 그중 환불은 나간 수."""
    stopped = refunded_no_reply = 0
    for f in FAILURES:
        _, refunded, replied = flow_with_refund_failures(f, cap)
        if not replied:
            stopped += 1
            refunded_no_reply += refunded
    return {'stopped': stopped, 'refunded_but_not_told': refunded_no_reply}


def report():
    out = {}
    out['serial'] = run_flow(serial_flow())
    out['parallel'] = run_flow(parallel_flow())
    out['serial_with_approval_120s'] = run_flow(serial_flow(), {**PARAMS, 'approval_wait_s': 120.0})
    out['cost_units_p_in1_p_out4'] = {
        'serial': cost_units(out['serial'], 1, 4), 'parallel': cost_units(out['parallel'], 1, 4)}
    normal = out['serial']
    loop = {}
    for cap in (10, 20, 50):
        calls, _, replied = flow_with_refund_failures(None, cap)
        r = run_flow(calls)
        loop[cap] = {**r, 'done': replied, 'input_x_normal': round(r['input_tokens'] / normal['input_tokens'], 2)}
    out['endless_error_loop'] = loop
    out['stopped_by_cap'] = {cap: stopped_count(cap) for cap in CAPS}
    return out


if __name__ == '__main__':
    import json
    print(json.dumps(report(), ensure_ascii=False, indent=1))
