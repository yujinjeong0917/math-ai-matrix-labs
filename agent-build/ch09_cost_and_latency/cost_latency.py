"""9장: 문의 한 건의 호출 수, 입력 누적, 대기 시간을 센다. 전부 가상 값이고 단가는 입력 변수예요."""

# (모듈 설명) 입력: 아래 PARAMS·TOOLS의 가상 값과 호출 목록. 출력: 호출 수, 입력·출력 토큰 합, 대기 시간(초).
# README 표와의 대응: 직렬 흐름 5호출, 입력 6520, 출력 280, 13.008초 / 병렬 흐름 3호출, 입력 4260, 10.604초.
# 토큰은 모델이 읽고 쓰는 글의 단위. 프롬프트 캐싱과 스트리밍은 반영하지 않는다.
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
        # 핵심: 도구를 한 번 부를 때마다 시스템 프롬프트와 지금까지의 이력 전체가 입력으로 다시 들어간다. 그래서 호출 수보다 입력 토큰이 더 빨리 늘어난다.
        input_tokens = params['system_tokens'] + history      # 이력을 통째로 다시 보내요
        total_in += input_tokens
        total_out += call['out']
        # 모델 대기 = 고정 0.5초 + 입력 x 0.0004초 + 출력 x 0.02초 (가상 계수)
        seconds += (params['model_base_s'] + input_tokens * params['model_s_per_in_token']
                    + call['out'] * params['model_s_per_out_token'])
        history += call['out']
        waits = [TOOLS[t][1] for t in call['tools']]
        if waits:
            # 병렬이면 가장 느린 도구만큼, 직렬이면 모두 더한 만큼 기다린다.
            seconds += max(waits) if call.get('parallel') else sum(waits)
        # 도구 결과도 이력에 쌓여 다음 호출의 입력이 된다. 직렬 5호출의 이력은 60, 250, 410, 850, 950으로 늘어난다(테스트의 손계산).
        history += sum(TOOLS[t][0] for t in call['tools'])
    return {'calls': len(calls), 'input_tokens': total_in, 'output_tokens': total_out,
            'seconds': round(seconds, 3)}


# 단가는 호출하는 쪽이 넣는 값이라 이 코드는 실제 요금을 말하지 않는다.
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


# 오류 재시도 모형: 조회 3번 뒤 환불을 failures번 실패시킨 다음 성공시킨다. 호출마다 이력이 쌓여 입력이 커진다.
def flow_with_refund_failures(failures, cap=None):
    """환불 도구가 failures번 오류를 돌려준 뒤 성공해요. failures=None이면 계속 오류예요.
    cap은 모델 호출 단계 상한(max steps)이에요.
    돌려주는 값: (호출 목록, 환불이 나갔나, 고객 안내까지 끝났나)."""
    calls = [{'out': STEP_OUT, 'tools': ['get_order']}, {'out': STEP_OUT, 'tools': ['get_customer']},
             {'out': STEP_OUT, 'tools': ['get_policy']}]
    n = 0
    while True:
        if cap is not None and len(calls) >= cap:
            # 단계 상한에 걸려 환불도 못 하고 멈춤
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
# 20건: 오류 0회 12건, 1회 4건, 2회 2건, 5회 1건, 끝까지 오류 1건(README의 분포)
FAILURES = [0] * 12 + [1] * 4 + [2] * 2 + [5] + [None]
CAPS = [4, 5, 6, 8, 10, 20]


def stopped_count(cap):
    """상한 cap에서 안내까지 가지 못하고 멈춘 문의 수와, 그중 환불은 나간 수."""
    stopped = refunded_no_reply = 0
    for f in FAILURES:
        # 함정: 환불은 나갔는데 안내할 단계가 없어 멈추면 고객은 환불 사실을 모른다(refunded_but_not_told).
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
    # 정상 문의(입력 6520)를 기준으로 계속 오류가 나는 문의의 입력 토큰을 배수로 비교한다(10단계 2.51배, 20단계 6.45배, 50단계 25.64배).
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
