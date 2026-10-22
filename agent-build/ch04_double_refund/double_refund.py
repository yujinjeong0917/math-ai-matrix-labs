import itertools


# 환불 서버를 흉내 낸 가짜예요. 실제 결제 서버 대신 아래 규칙대로만 움직여요.
class RefundServer:
    def __init__(self, refund_amounts):
        self.refund_amounts = refund_amounts                                # 주문번호 -> 환불 금액
        self.order_status = {n: '결제됨' for n in refund_amounts}            # 주문번호 -> 지금 상태
        self.payouts = []                                                   # 실제로 나간 환불 목록 (주문번호, 금액)
        self.saved_results = {}                                             # 멱등성 키 -> 처음 처리한 결과

    def refund(self, order_id, idempotency_key=None, reply_is_late=False):
        # 멱등성 키(Idempotency-Key)는 "같은 요청이에요"라고 알리는 번호표예요.
        if idempotency_key is not None and idempotency_key in self.saved_results:
            return self.saved_results[idempotency_key]   # 이미 본 번호표: 처음 결과만 돌려주고 새로 지급하지 않아요
        self.order_status[order_id] = '환불됨'                                # 상태를 바꾸고
        self.payouts.append((order_id, self.refund_amounts[order_id]))      # 금액을 지급 목록에 적어요
        if idempotency_key is not None:
            self.saved_results[idempotency_key] = 'ok'                      # 번호표와 결과를 저장해 둬요
        if reply_is_late:
            raise TimeoutError   # 처리는 끝났는데 응답이 늦은 경우: 보낸 쪽은 성공인지 실패인지 "모름"
        return 'ok'


# 주문 1번부터 10번까지의 환불 금액 (합계 330 = 정답)
refund_amounts = {n: v for n, v in enumerate([30, 20, 50, 40, 25, 60, 15, 35, 45, 10], start=1)}
LATE_REPLY_ORDERS = {3, 7}          # 첫 시도에서 응답이 늦는 주문
DUPLICATE_MESSAGE_ORDERS = {5}      # 고객이 같은 메시지를 두 번 보내는 주문 (이중 전송)


def simulate(variant):
    # 'A' 방어 없음, 'B' 보내기 전에 상태 조회, 'C' 주문마다 같은 번호표, 'D' 시도마다 새 번호표
    server = RefundServer(refund_amounts)
    new_key_numbers = itertools.count(1)

    def idempotency_key_for(order_id):
        if variant == 'C':
            return f'refund-{order_id}'               # 같은 주문이면 늘 같은 번호표
        if variant == 'D':
            return f'new-{next(new_key_numbers)}'     # 부를 때마다 새 번호표
        return None                                   # A와 B는 번호표 없음

    for order_id in refund_amounts:
        if order_id in DUPLICATE_MESSAGE_ORDERS:
            # 같은 메시지 두 개: 상태 조회를 둘 다 먼저 하고, 환불 실행을 둘 다 해요 (동시 실행은 아니에요)
            first_check = second_check = (variant != 'B') or server.order_status[order_id] == '결제됨'
            for may_send in (first_check, second_check):
                if may_send:
                    try:
                        server.refund(order_id, idempotency_key_for(order_id))
                    except TimeoutError:
                        pass
            continue
        for attempt in range(3):     # 처음 한 번 + 응답이 없으면 최대 2번 다시 보내요
            if variant == 'B' and server.order_status[order_id] != '결제됨':
                break                # B: 이미 환불됐으면 다시 보내지 않고 멈춰요
            try:
                server.refund(order_id, idempotency_key_for(order_id),
                              reply_is_late=(attempt == 0 and order_id in LATE_REPLY_ORDERS))
                break                # 응답이 왔으면 끝
            except TimeoutError:
                continue             # 응답이 없으면 다시 보내요
    return server


def run_variant(variant):
    # 돌려주는 세 값: 환불이 실행된 횟수, 지급 합계, 정답(330)보다 더 나간 금액
    server = simulate(variant)
    total_paid = sum(amount for _, amount in server.payouts)
    return len(server.payouts), total_paid, total_paid - sum(refund_amounts.values())


if __name__ == '__main__':
    for variant in 'ABCD':
        print(variant, run_variant(variant))
