"""0장: 환불 문의 한 건을 단계로 풀어 도구 호출 수, 쓰기 호출 수, 되돌릴 수 없는 호출 수를 센다.

가짜 모델도 가짜 서버도 쓰지 않는 단순 집계예요. 문의 20건(가상)마다 필요한 도구 호출 순서를
inquiries.json에 고정해 두었고, 이 파일은 그 목록을 세기만 해요. 난수가 없어서 몇 번을 돌려도 같아요.

입력: inquiries.json (문의 id, 종류, 도구 호출 순서). 출력: 집계 딕셔너리와 문의별 행.
README 기대 출력과의 대응: cases 20, total_calls 96, total_writes 53, total_irreversible 30,
total_money_out 10, 환불 문의만의 평균 호출 수 61/12(약 5.083). 모두 이 파일이 세는 값이다.
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
    """문의 한 건의 호출 수, 읽기/쓰기 수, 되돌릴 수 없는 호출 수, 처음 되돌릴 수 없는 호출의 순번을 센다."""
    calls = inquiry["calls"]
    kinds = [TOOLS[c]["kind"] for c in calls]
    # i는 1부터 시작하는 호출 순번. 순번 값이 곧 "그 호출 직전까지 몇 번 호출했나 + 1"이 된다.
    # 쓰기이면서 되돌릴 수 없는 호출만 센다(읽기는 reversible이 True라 조건과 무관하게 제외).
    irreversible = [i for i, c in enumerate(calls, start=1) if TOOLS[c]["kind"] == "write" and not TOOLS[c]["reversible"]]
    # 접수 단계는 도구 호출 없이 항상 거치므로 집합에 미리 넣는다(모형의 선택).
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
        # 돈이 실제로 나간 횟수 = issue_refund 호출 수
        "money_out": calls.count("issue_refund"),
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def summarize(inquiries=None):
    rows = [profile(q) for q in (inquiries or load_inquiries())]
    # 평균은 분모를 무엇으로 잡느냐에 따라 달라진다. 환불 문의(12건)와 전체(20건)를 구분해 둔다.
    refunds = [r for r in rows if r["kind"] == "환불"]
    # 환불 문의 중 실제로 돈이 나간 건(9건). 첫 되돌릴 수 없는 호출 순번의 평균을 낼 때 None을 피하려고 따로 거른다.
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
        # 순번 - 1 = 첫 되돌릴 수 없는 호출 앞에 놓인 호출 수. 테스트의 기대값 21/9와 대응(순번 3,3,4,3,4,3,4,3,3).
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
