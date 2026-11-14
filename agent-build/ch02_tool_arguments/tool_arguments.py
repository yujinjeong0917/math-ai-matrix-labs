"""2장: 도구 호출 값 30개(고정 목록)를 두 단계로 검사해, 누가 무엇을 잡는지 센다.

1단계 스키마 검사: 필수 값, 자료형, 허용 목록, 주문번호 모양처럼 값만 보고 알 수 있는 것.
2단계 의미 검사: 주문 DB(가상)와 세션 고객을 보고 알 수 있는 것(없는 주문, 남의 주문, 금액 초과).
"의도"(intended)는 고객이 실제로 원한 값이에요. 코드가 알 수 없는 정답이라 채점에만 써요.
모델은 쓰지 않고, 모델이 만들었다고 치는 호출을 고정해 두었어요.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SESSION_CUSTOMER = "C-100"

# 주문 DB(가상). paid: 결제 금액(원), refunded: 이미 환불된 금액(원)
ORDERS = {
    "ORD-1001": {"customer": "C-100", "paid": 50000, "refunded": 0},
    "ORD-1002": {"customer": "C-100", "paid": 30000, "refunded": 0},
    "ORD-1003": {"customer": "C-100", "paid": 45000, "refunded": 15000},
    "ORD-1004": {"customer": "C-100", "paid": 12000, "refunded": 0},
    "ORD-1005": {"customer": "C-100", "paid": 89000, "refunded": 0},
    "ORD-1006": {"customer": "C-100", "paid": 25000, "refunded": 0},
    "ORD-2001": {"customer": "C-200", "paid": 70000, "refunded": 0},
    "ORD-2002": {"customer": "C-200", "paid": 18000, "refunded": 0},
}

# issue_refund 도구의 입력 스키마(JSON Schema의 일부만 흉내 낸 최소 검사기용)
SCHEMA = {
    "required": ["order_id", "amount", "reason"],
    "properties": {
        "order_id": {"type": "string", "pattern": r"^ORD-\d{4}$"},
        "amount": {"type": "integer", "minimum": 1},
        "reason": {"type": "string", "enum": ["단순변심", "불량", "오배송", "기타"]},
    },
    "additionalProperties": False,
}


def load_calls():
    return json.loads((HERE / "calls.json").read_text(encoding="utf-8"))["calls"]


def _is_type(value, name):
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    raise ValueError(name)


def schema_check(call, schema=SCHEMA):
    errors = []
    for key in schema["required"]:
        if key not in call:
            errors.append(f"필수 값 없음: {key}")
    for key, value in call.items():
        spec = schema["properties"].get(key)
        if spec is None:
            if not schema["additionalProperties"]:
                errors.append(f"정의에 없는 값: {key}")
            continue
        if not _is_type(value, spec["type"]):
            errors.append(f"자료형 틀림: {key}")
            continue
        if "pattern" in spec and not re.match(spec["pattern"], value):
            errors.append(f"모양 틀림: {key}")
        if "minimum" in spec and value < spec["minimum"]:
            errors.append(f"최솟값 미만: {key}")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"허용 목록에 없음: {key}")
    return errors


def semantic_check(call, session_customer=SESSION_CUSTOMER, orders=ORDERS):
    order = orders.get(call["order_id"])
    if order is None:
        return ["없는 주문"]
    errors = []
    if order["customer"] != session_customer:
        errors.append("다른 고객의 주문")
    if call["amount"] > order["paid"] - order["refunded"]:
        errors.append("환불 가능 금액 초과")
    return errors


def classify(item):
    call, intended = item["call"], item["intended"]
    schema_errors = schema_check(call)
    semantic_errors = [] if schema_errors else semantic_check(call)
    matches_intent = call == intended
    if schema_errors:
        outcome = "형식 오류로 걸림" if not matches_intent else "정상인데 거절됨"
    elif semantic_errors:
        outcome = "의미 오류로 걸림" if not matches_intent else "정상인데 거절됨"
    else:
        outcome = "통과(정상)" if matches_intent else "조용한 오류"
    return {"id": item["id"], "outcome": outcome, "schema_errors": schema_errors, "semantic_errors": semantic_errors}


def summarize(items=None):
    rows = [classify(i) for i in (items or load_calls())]
    counts = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    wrong = sum(v for k, v in counts.items() if k not in ("통과(정상)", "정상인데 거절됨"))
    return {
        "calls": len(rows),
        "counts": counts,
        "wrong_calls": wrong,
        "wrong_passed_schema_only": wrong - counts.get("형식 오류로 걸림", 0),  # 의미 검사가 없을 때 실행될 잘못된 호출
        "wrong_passed_both": counts.get("조용한 오류", 0),
        "silent_ids": [r["id"] for r in rows if r["outcome"] == "조용한 오류"],
        "rows": rows,
    }


if __name__ == "__main__":
    s = summarize()
    print("호출", s["calls"], "잘못된 호출", s["wrong_calls"])
    for k, v in s["counts"].items():
        print(" ", k, v)
    print("의미 검사 없이 실행될 잘못된 호출", s["wrong_passed_schema_only"])
    print("조용한 오류의 호출 번호", s["silent_ids"])
