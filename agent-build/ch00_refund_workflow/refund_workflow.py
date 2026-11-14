"""0장: 환불 문의 한 건을 단계로 풀어 도구 호출 수, 쓰기 호출 수, 되돌릴 수 없는 호출 수를 센다.

가짜 모델도 가짜 서버도 쓰지 않는 단순 집계예요. 문의 20건(가상)마다 필요한 도구 호출 순서를
inquiries.json에 고정해 두었고, 이 파일은 그 목록을 세기만 해요. 난수가 없어서 몇 번을 돌려도 같아요.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 도구 목록(가상). kind: 읽기/쓰기, reversible: 호출한 뒤 되돌릴 수 있는가, stage: 어느 단계의 도구인가
TOOLS = {
    "lookup_order":    {"kind": "read",  "reversible": True,  "stage": "조회"},
    "get_tracking":    {"kind": "read",  "reversible": True,  "stage": "조회"},
    "check_stock":     {"kind": "read",  "reversible": True,  "stage": "조회"},
    "get_policy":      {"kind": "read",  "reversible": True,  "stage": "판단"},
    "issue_refund":    {"kind": "write", "reversible": False, "stage": "실행"},
    "create_exchange": {"kind": "write", "reversible": True,  "stage": "실행"},  # 출고 전 취소 가능하다고 가정(가상)
    "send_message":    {"kind": "write", "reversible": False, "stage": "안내"},  # 보낸 메시지는 거둬들일 수 없다
    "log_note":        {"kind": "write", "reversible": True,  "stage": "기록"},  # 메모는 고쳐 쓸 수 있다
}


def load_inquiries():
    return json.loads((HERE / "inquiries.json").read_text(encoding="utf-8"))["inquiries"]


def profile(inquiry):
    calls = inquiry["calls"]
    kinds = [TOOLS[c]["kind"] for c in calls]
    irreversible = [i for i, c in enumerate(calls, start=1) if TOOLS[c]["kind"] == "write" and not TOOLS[c]["reversible"]]
    stages = {"접수"} | {TOOLS[c]["stage"] for c in calls}
    return {
        "id": inquiry["id"],
        "kind": inquiry["kind"],
        "calls": len(calls),
        "reads": kinds.count("read"),
        "writes": kinds.count("write"),
        "irreversible": len(irreversible),
        "first_irreversible_at": irreversible[0] if irreversible else None,
        "stages": len(stages),
        "money_out": calls.count("issue_refund"),
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def summarize(inquiries=None):
    rows = [profile(q) for q in (inquiries or load_inquiries())]
    refunds = [r for r in rows if r["kind"] == "환불"]
    paid = [r for r in refunds if r["money_out"] > 0]
    return {
        "cases": len(rows),
        "by_kind": {k: sum(1 for r in rows if r["kind"] == k) for k in ("환불", "교환", "배송조회")},
        "total_calls": sum(r["calls"] for r in rows),
        "total_writes": sum(r["writes"] for r in rows),
        "total_irreversible": sum(r["irreversible"] for r in rows),
        "total_money_out": sum(r["money_out"] for r in rows),
        # 환불 문의 20건이라고 잘못 세었을 때의 평균(깨지는 장면)
        "naive_mean_calls_all20": mean(r["calls"] for r in rows),
        "refund_mean_calls": mean(r["calls"] for r in refunds),
        "refund_mean_writes": mean(r["writes"] for r in refunds),
        "refund_mean_irreversible": mean(r["irreversible"] for r in refunds),
        "naive_mean_irreversible_all20": mean(r["irreversible"] for r in rows),
        "paid_refund_cases": len(paid),
        "paid_first_irreversible_at": [r["first_irreversible_at"] for r in paid],
        "paid_mean_calls": mean(r["calls"] for r in paid),
        "paid_mean_reads_before_first_irreversible": mean(r["first_irreversible_at"] - 1 for r in paid),
        "mean_stages_refund": mean(r["stages"] for r in refunds),
        "rows": rows,
    }


if __name__ == "__main__":
    s = summarize()
    for k, v in s.items():
        if k != "rows":
            print(k, v)
    for r in s["rows"]:
        print(r["id"], r["kind"], r["calls"], r["reads"], r["writes"], r["irreversible"], r["first_irreversible_at"])
