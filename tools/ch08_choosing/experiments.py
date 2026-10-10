"""AI 도구 실전 8장 실험. `uv run python tools/ch08_choosing/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 한 주 비용 x 52 / 12, 모델만 바꿀 때의 폭
E1 1~7장 한 달 비용 표(행마다 센 것과 단위가 다름, 6·7장은 결과 대기)
E2 데이터 처리 약관 표 요약(학습 기본값, 보관, 지역)
E3 프롬프트 주입 장난감 2판: 고리 6가지, 5장 결과와 이어 보기
E4 예시 계산: 1년 동안 숨은 지시가 피해로 이어지는 기대 횟수 N x p x q, 확인 문 질문 수
E5 선택 점검표: 후보 구성 4개 x 질문 10개
E6 결정성과 대조: 데이터 재생성, 두 번 실행, 장난감 결과를 문의 표에서 직접 센 값과 비교

표준 라이브러리만 쓴다. 네트워크·API 키가 필요 없고 결과는 매번 같다.
"""

import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import tools_ch08_checklist as K  # noqa: E402
import tools_ch08_cost as C  # noqa: E402
import tools_ch08_data as D  # noqa: E402
import tools_ch08_injection as I  # noqa: E402

OUT = HERE / "results" / "ch08.json"


def e2_privacy():
    pol = D.load_policies()
    rows = [{k: r[k] for k in ("provider", "product", "tier", "trains_default", "retention", "region", "url", "checked")}
            for r in pol["rows"]]
    need_check = [r["provider"] + " " + r["product"] for r in pol["rows"] if "[확인 필요]" in (r["note"] or "")]
    by = {}
    for r in pol["rows"]:
        by[r["trains_default"]] = by.get(r["trains_default"], 0) + 1
    return {"rows": rows, "trains_default_counts": by, "need_check": need_check, "n_rows": len(rows)}


def e3_injection():
    up = D.load_upstream()
    sc = I.all_scenarios()
    return {"ch05_naive_with_write": {"notion_rows": up["ch05_naive_write_notion_rows"],
                                      "figma_texts": up["ch05_naive_write_figma"]},
            "user_task": I.USER_TASK, "filter_words": I.FILTER_WORDS, "leak_kinds_assumed": list(I.LEAK_KINDS),
            "scenarios": sc}


def e4_yearly():
    y = I.yearly()
    d = next(s for s in I.all_scenarios() if s["key"] == "D")
    e = next(s for s in I.all_scenarios() if s["key"] == "E")
    y["prompts_week_per_call"] = d["approval_prompts"]
    y["prompts_year_per_call"] = d["approval_prompts"] * 52
    y["prompts_week_per_kind_E"] = e["approval_prompts"]
    y["toy_leak_ratio"] = f"{len(I.LEAK_KINDS)}/4"
    return y


def e6_determinism():
    inbox_same = D.make_inbox() == D.load_inbox()
    upstream_same = D.build_upstream()["values"] == json.loads(D.UPSTREAM_FILE.read_text(encoding="utf-8"))["values"]
    twice = I.all_scenarios() == I.all_scenarios()
    rows = D.load_inbox()["rows"]
    direct = {"rows_not_done": sum(1 for r in rows if r["status"] != "답변 완료"),
              "emails": len({r["email"] for r in rows}),
              "marked_injections": sorted(r["id"] for r in rows if r["_mark"] and r["_mark"].startswith("I")),
              "marked_false_positive": [r["id"] for r in rows if r["_mark"] == "FP"]}
    a = next(s for s in I.all_scenarios() if s["key"] == "A")
    agree = (a["notion_rows_changed"] == direct["rows_not_done"] and a["emails_leaked"] == direct["emails"]
             and sorted(x[0] for x in a["followed"]) == direct["marked_injections"])
    return {"inbox_regenerates": inbox_same, "upstream_snapshot_matches_live": upstream_same,
            "scenarios_twice_identical": twice, "direct": direct, "toy_agrees_with_direct_count": agree}


def main():
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "seed": None, "policy_file": D.POLICY_FILE.name, "upstream_file": D.UPSTREAM_FILE.name,
                 "checked": D.CHECKED, "pending_upstream": []},
        "e0_hand": C.hand(),
        "e1_cost": C.table(),
        "e2_privacy": e2_privacy(),
        "e3_injection": e3_injection(),
        "e4_yearly": e4_yearly(),
        "e5_checklist": K.evaluate(),
        "e6_determinism": e6_determinism(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
