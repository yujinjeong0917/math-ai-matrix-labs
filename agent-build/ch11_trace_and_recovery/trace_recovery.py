"""11장: 장애 6가지에서 원인 찾기 단계 수와 복구 경로를 세고, 4장 변형 C를 동시 요청 순서 전부에 돌린다.
확인 단계 수와 복구 규칙은 이 장난감이 정한 가정이에요. 실제 결제 서비스의 취소·정정 방식은 재지 않아요."""
import itertools

STAGES = ['접수', '분류', '조회', '판단', '실행', '안내', '기록']   # 1~7번

# 장애: (이름, 증상이 눈에 띈 단계, 실제 원인이 있는 단계)
# 증상은 고객 문의가 다시 오거나(안내 단계) 정산표를 볼 때(기록 단계) 드러난다고 가정해요.
FAILURES = [
    ('시간 초과 뒤 재시도로 이중 환불', 7, 5),
    ('결제사의 일시 거절', 6, 5),
    ('조회 뒤 주문 상태가 바뀜', 7, 3),
    ('판단 단계의 금액 계산 오류(10 적게 지급)', 7, 4),
    ('교환 문의를 환불로 분류', 6, 2),
    ('환불은 됐는데 고객 안내 누락', 6, 6),
]


def steps_to_find_cause(symptom_stage, origin_stage, has_trace):
    """기록이 없으면 증상이 보인 단계에서 거꾸로 한 곳씩 연다(가정). 요청 번호가 있는 기록이 있으면 한 번 조회해요."""
    return 1 if has_trace else symptom_stage - origin_stage + 1


def find_summary():
    without = [steps_to_find_cause(s, o, False) for _, s, o in FAILURES]
    with_trace = [steps_to_find_cause(s, o, True) for _, s, o in FAILURES]
    return {'without_trace': without, 'with_trace': with_trace,
            'sum_without': sum(without), 'sum_with': sum(with_trace)}


# 복구 경로 규칙(가정). 같은 키 재시도와 안내 재발송은 돈을 새로 움직이지 않는 일이에요.
RECOVERY = ['자동 재시도', '보상 입금', '사람 인계']
ROUTE_WITH_REQUEST_ID = {
    '시간 초과 뒤 재시도로 이중 환불': '사람 인계',       # 이미 나간 돈 회수는 사람이 판단
    '결제사의 일시 거절': '자동 재시도',                    # 같은 키로 다시 시도
    '조회 뒤 주문 상태가 바뀜': '사람 인계',
    '판단 단계의 금액 계산 오류(10 적게 지급)': '보상 입금',  # 모자란 10을 요청 번호에 묶어 추가 지급
    '교환 문의를 환불로 분류': '사람 인계',
    '환불은 됐는데 고객 안내 누락': '자동 재시도',          # 안내 재발송
}
SETTLEMENT = [30, 20, 50, 40, 25, 30, 15, 35, 30, 10, 45, 30]   # 가상 정산표 12행(지급 금액)
WRONGLY_PAID = 30   # 잘못 지급된 금액(40을 줘야 하는데 30을 준 건)


def route(failure_name, has_request_id):
    """요청 번호가 기록에 없으면 어느 건인지 몰라 자동 재시도와 보상 입금은 쓸 수 없고 모두 사람에게 가요."""
    return ROUTE_WITH_REQUEST_ID[failure_name] if has_request_id else '사람 인계'


def recovery_summary(has_request_id):
    counts = {r: 0 for r in RECOVERY}
    for name, _, _ in FAILURES:
        counts[route(name, has_request_id)] += 1
    return counts


def compensation_candidates(has_request_id):
    """보상 입금 대상으로 의심되는 정산표 행의 수. 번호가 없으면 금액만 보고 찾아요."""
    return 1 if has_request_id else sum(1 for a in SETTLEMENT if a == WRONGLY_PAID)


# ---- 4장 변형 C를 동시 요청에 돌리기 ----
# 4장의 서버는 (1) 키를 이미 봤나 확인 (2) 환불 실행 (3) 키와 결과 저장을 차례로 해요.
# 같은 키의 요청 둘이 이 세 걸음을 어떤 순서로든 섞어 실행할 때 지급이 몇 번 나가는지 모든 순서를 센다.

def interleavings():
    """요청 a, b가 각각 걸음 3개를 가질 때 모든 섞는 순서(안쪽 순서 유지): 20가지."""
    for a_slots in itertools.combinations(range(6), 3):
        order, ai, bi = [], 0, 0
        for pos in range(6):
            if pos in a_slots:
                order.append(('a', ai)); ai += 1
            else:
                order.append(('b', bi)); bi += 1
        yield order


def run_check_then_store(order):
    """4장 방식: 확인 / 실행 / 저장이 따로 일어나요."""
    saved, payouts = set(), 0
    seen = {}
    for who, step in order:
        if step == 0:
            seen[who] = 'key' in saved
        elif step == 1 and not seen[who]:
            payouts += 1
        elif step == 2 and not seen[who]:
            saved.add('key')
    return payouts


def run_atomic_reserve(order):
    """고친 방식: 첫 걸음에서 확인과 자리 잡기를 한 번에 해요. 자리가 잡혀 있으면 이번 요청은 충돌로 돌려보내고 지급하지 않아요."""
    reserved, payouts = False, 0
    mine = {}
    for who, step in order:
        if step == 0:
            mine[who] = not reserved
            reserved = True
        elif step == 1 and mine[who]:
            payouts += 1
        # step 2(결과 저장)는 지급 횟수에 영향이 없어요
    return payouts


def race_summary():
    orders = list(interleavings())
    plain = [run_check_then_store(o) for o in orders]
    atomic = [run_atomic_reserve(o) for o in orders]
    return {'orders': len(orders),
            'check_then_store_double_pay_orders': sum(p > 1 for p in plain),
            'check_then_store_max_payouts': max(plain),
            'atomic_double_pay_orders': sum(p > 1 for p in atomic),
            'atomic_max_payouts': max(atomic),
            'atomic_zero_pay_orders': sum(p == 0 for p in atomic)}


def report():
    return {'find': find_summary(),
            'recovery_with_request_id': recovery_summary(True),
            'recovery_without_request_id': recovery_summary(False),
            'compensation_candidates_with_id': compensation_candidates(True),
            'compensation_candidates_without_id': compensation_candidates(False),
            'race': race_summary()}


if __name__ == '__main__':
    import json
    print(json.dumps(report(), ensure_ascii=False, indent=1))
