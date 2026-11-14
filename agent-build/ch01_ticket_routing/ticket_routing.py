"""1장: 문의 40건(가상)을 키워드 규칙으로 분류하고, 혼동행렬과 오분류 비용(가상)을 센다.

가짜 분류기는 키워드 규칙 세 가지(v1, v2, v3)뿐이고 실제 언어 모델이 아니에요. 난수가 없어요.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = ["환불", "교환", "배송조회", "기타", "사람확인"]  # 정답에는 앞의 넷, 예측에는 다섯 모두 쓴다

# 가상 비용표(원). 행: 정답, 열: 예측. 환불이 잘못 실행되면 되돌릴 수 없어서 가장 커요.
COST = {
    "환불":    {"환불": 0, "교환": 6000,  "배송조회": 4000, "기타": 3000, "사람확인": 1500},
    "교환":    {"환불": 60000, "교환": 0, "배송조회": 4000, "기타": 3000, "사람확인": 1500},
    "배송조회": {"환불": 60000, "교환": 4000, "배송조회": 0, "기타": 3000, "사람확인": 1500},
    "기타":    {"환불": 60000, "교환": 4000, "배송조회": 4000, "기타": 0, "사람확인": 1500},
}

NEGATED_REFUND = re.compile(r"환불\s*(말고|아니|은 필요 없|이 아니)")


def load_tickets():
    return json.loads((HERE / "tickets.json").read_text(encoding="utf-8"))["tickets"]


def route_v1(text):
    """환불 낱말을 가장 먼저 보는 우선순위 규칙이에요. 환불 -> 교환 -> 배송 -> 기타 순서."""
    if "환불" in text:
        return "환불"
    if "교환" in text:
        return "교환"
    if re.search(r"배송|도착|송장|택배", text):
        return "배송조회"
    return "기타"


def route_v2(text):
    """'환불 말고' 같은 부정 표현이 있으면 환불로 보지 않아요."""
    if "환불" in text and not NEGATED_REFUND.search(text):
        return "환불"
    if re.search(r"교환|바꿔", text):
        return "교환"
    if re.search(r"배송|도착|송장|택배", text):
        return "배송조회"
    return "기타"


def route_v3(text):
    """환불과 교환 낱말이 같이 있으면 맞히려 하지 않고 사람에게 넘겨요. 나머지는 v2와 같아요."""
    if "환불" in text and "교환" in text:
        return "사람확인"
    return route_v2(text)


ROUTERS = {"v1": route_v1, "v2": route_v2, "v3": route_v3}


def evaluate(router):
    matrix = {t: {p: 0 for p in LABELS} for t in LABELS[:4]}
    wrong_refund_executions = []
    cost = 0
    correct = 0
    for ticket in load_tickets():
        pred = router(ticket["text"])
        matrix[ticket["label"]][pred] += 1
        cost += COST[ticket["label"]][pred]
        correct += pred == ticket["label"]
        if pred == "환불" and ticket["label"] != "환불":
            wrong_refund_executions.append(ticket["id"])
    predicted_refund = sum(matrix[t]["환불"] for t in matrix)
    true_refund = sum(matrix["환불"].values())
    return {
        "matrix": matrix,
        "refund_predicted": predicted_refund,
        "refund_precision": matrix["환불"]["환불"] / predicted_refund,
        "refund_recall": matrix["환불"]["환불"] / true_refund,
        "correct": correct,
        "handoff": sum(matrix[t]["사람확인"] for t in matrix),
        "cost": cost,
        "wrong_refund_executions": wrong_refund_executions,
        "missed_refunds": [t["id"] for t in load_tickets() if t["label"] == "환불" and router(t["text"]) != "환불"],
    }


if __name__ == "__main__":
    for name, router in ROUTERS.items():
        r = evaluate(router)
        print(name, "정답", r["correct"], "/ 40", "사람확인", r["handoff"], "비용", r["cost"])
        print("  잘못 실행된 환불", len(r["wrong_refund_executions"]), r["wrong_refund_executions"])
        print("  환불 정밀도", f"{r['matrix']['환불']['환불']}/{r['refund_predicted']}", round(r["refund_precision"], 3),
              "재현율", f"{r['matrix']['환불']['환불']}/13", round(r["refund_recall"], 3))
        print("  놓친 환불", len(r["missed_refunds"]), r["missed_refunds"])
        for t, row in r["matrix"].items():
            print("  ", t, [row[p] for p in LABELS])
