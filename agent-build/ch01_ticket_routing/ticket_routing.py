"""1장: 문의 40건(가상)을 키워드 규칙으로 분류하고, 혼동행렬과 오분류 비용(가상)을 센다.

가짜 분류기는 키워드 규칙 세 가지(v1, v2, v3)뿐이고 실제 언어 모델이 아니에요. 난수가 없어요.

입력: tickets.json (문의 40건: id, 본문, 정답 라벨). 출력: 규칙별 정답 수, 사람 확인 수, 비용, 잘못 실행된 환불.
README 표와의 대응: v1 29/0/8/490000, v2 31/0/5/313000, v3 28/6/2/142000, 환불 정밀도·재현율 11/19·11/13 등.
비용(원)은 가상 값이다. 실제 손실을 조사한 값이 아니다.
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

# "환불 말고", "환불은 필요 없" 같은 부정 표현. 단순 키워드 규칙이라 "환불 말고는 방법이 없네요, 환불 진행해 주세요"처럼 뒤에 환불을 다시 요청하는 문장도 부정으로 오인한다(t13).
NEGATED_REFUND = re.compile(r"환불\s*(말고|아니|은 필요 없|이 아니)")


def load_tickets():
    return json.loads((HERE / "tickets.json").read_text(encoding="utf-8"))["tickets"]


def route_v1(text):
    """환불 낱말을 가장 먼저 보는 우선순위 규칙이에요. 환불 -> 교환 -> 배송 -> 기타 순서."""
    # 규칙 v1의 약점: 낱말이 있기만 하면 환불로 보내 "환불 말고 교환"도 환불이 된다.
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
    # 둘 다 나오면 규칙이 틀릴 가능성이 높아, 정답을 포기하고 사람에게 넘긴다. 정확도(정답 수)는 내려가고 비용은 내려간다.
    if "환불" in text and "교환" in text:
        return "사람확인"
    return route_v2(text)


ROUTERS = {"v1": route_v1, "v2": route_v2, "v3": route_v3}


def evaluate(router):
    """규칙 하나를 40건에 돌려 혼동행렬, 정답 수, 비용, 잘못 실행된 환불 id, 놓친 환불 id를 돌려준다."""
    # 혼동행렬: 행은 정답(앞 네 라벨), 열은 예측(다섯 라벨). "사람확인"은 예측에만 있다.
    matrix = {t: {p: 0 for p in LABELS} for t in LABELS[:4]}
    wrong_refund_executions = []
    cost = 0
    correct = 0
    for ticket in load_tickets():
        pred = router(ticket["text"])
        matrix[ticket["label"]][pred] += 1
        # COST[정답][예측]. 비용표는 행이 정답이라 정답/예측 순서를 바꿔 읽으면 값이 어긋난다.
        cost += COST[ticket["label"]][pred]
        # bool을 정수로 더한다(True는 1). 사람확인은 어떤 정답과도 같지 않아 오답으로 센다.
        correct += pred == ticket["label"]
        if pred == "환불" and ticket["label"] != "환불":
            wrong_refund_executions.append(ticket["id"])
    # 열 합: 환불이라고 예측한 건수. 정밀도의 분모(v1은 19).
    predicted_refund = sum(matrix[t]["환불"] for t in matrix)
    # 행 합: 실제 환불 건수 13. 재현율의 분모.
    true_refund = sum(matrix["환불"].values())
    return {
        "matrix": matrix,
        "refund_predicted": predicted_refund,
        # 정밀도 = 맞게 환불로 보낸 수 / 환불로 예측한 수, 재현율 = 맞게 환불로 보낸 수 / 실제 환불 수
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
