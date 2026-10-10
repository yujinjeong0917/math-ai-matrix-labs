"""AI 도구 실전 4장 실험. `uv run python tools/ch04_skills/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 스킬 N개, 앞머리 100토큰·본문 2,000토큰이라 치고 "전부 싣기"와 "앞머리만 + 쓰는 것만 본문"
E1 SKILL.md 앞머리 읽기: data/skills/ 6개를 파서로 읽고 사양·Anthropic 규칙으로 검사
E2 규칙을 어긴 앞머리: 사양 문서의 잘못된 예와 직접 만든 예
E3 파일별 근사 토큰 수: 목록 한 줄, 본문, 딸린 파일, 프로젝트 지침
E4 한 주(대화 15번)의 지시문 토큰: 전부 싣기 / 단계적 공개, 토큰 규칙 세 가지로 민감도
E5 스킬 수 N을 늘릴 때: 우리 파일 평균과 문서가 밝힌 대략값
E6 대신 치르는 값: 파일을 읽을 때마다 호출이 한 번 더 늘 때의 입력(캐싱 없음 / 2장 캐싱 배율)
E7 한 파일에 몰아 넣으면: UTF-8 바이트와 Codex의 32 KiB, 줄 수와 Claude Code의 200줄 권장
E8 대조: 무작위 스킬 묶음 300경우에서 닫힌 식 / 하나씩 세기, 단일 배율 규칙에서 비율이 그대로인지

표준 라이브러리만 쓴다. 고정 시드라 결과는 매번 같다.
"""

import json
import platform
import random
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import tools_ch04_budget as B  # noqa: E402
import tools_ch04_frontmatter as FM  # noqa: E402

OUT = HERE / "results" / "ch04.json"
SEED = 4


def f(x, nd=4):
    return round(float(x), nd)


def e0_hand():
    m, b = 100, 2000
    rows = []
    for n in (1, 2, 5):
        allt, st = n * (m + b), n * m + b
        rows.append({"n": n, "paste_all": allt, "staged": st, "ratio": f(F(st, allt)),
                     "saving_pct": f(100 * (1 - F(st, allt)), 1)})
    return {"meta": m, "body": b, "rows": rows, "limit_ratio": f(F(m, m + b)),
            "limit_saving_pct": f(100 * (1 - F(m, m + b)), 1)}


def e1_parse(skills):
    main = next(s for s in skills if s["name"] == "weekly-inquiry-report")
    line, cut = FM.listing_line(main["fm"])
    return {
        "main": {"fields": list(main["fm"].keys()), "name": main["fm"]["name"],
                 "name_len": len(main["fm"]["name"]), "description_len": len(main["fm"]["description"]),
                 "metadata": main["fm"]["metadata"], "body_lines": len(main["body"].strip().split("\n")),
                 "listing_line_truncated": cut, "listing_line_len": len(line)},
        "all": [{"dir": s["dir"], "name": s["name"], "description_len": len(s["fm"]["description"]),
                 "spec_errors": FM.validate_spec(s["fm"], s["dir"]),
                 "anthropic_errors": FM.validate_anthropic(s["fm"])} for s in skills],
    }


BAD = [
    ("대문자", "---\nname: PDF-Processing\ndescription: PDF를 다뤄요.\n---\n", "PDF-Processing"),
    ("하이픈으로 시작", "---\nname: -pdf\ndescription: PDF를 다뤄요.\n---\n", "-pdf"),
    ("하이픈 연속", "---\nname: pdf--processing\ndescription: PDF를 다뤄요.\n---\n", "pdf--processing"),
    ("폴더 이름과 다름", "---\nname: weekly-report\ndescription: 보고서를 써요.\n---\n", "weekly-inquiry-report"),
    ("설명 1,025자", "---\nname: long-desc\ndescription: " + "가" * 1025 + "\n---\n", "long-desc"),
    ("설명 없음", "---\nname: no-desc\n---\n본문만 있어요.\n", "no-desc"),
    ("예약어", "---\nname: claude-report\ndescription: 보고서를 써요.\n---\n", "claude-report"),
    ("XML 태그", "---\nname: tag-desc\ndescription: <b>보고서</b>를 써요.\n---\n", "tag-desc"),
    ("이름 65자", "---\nname: " + "a" * 65 + "\ndescription: 길어요.\n---\n", "a" * 65),
]


def e2_bad():
    rows = []
    for label, text, d in BAD:
        fm, _ = FM.parse(text)
        rows.append({"case": label, "spec_errors": FM.validate_spec(fm, d), "anthropic_errors": FM.validate_anthropic(fm)})
    try:
        FM.parse("name: x\ndescription: y\n")
        no_open = None
    except FM.FrontmatterError as e:
        no_open = str(e)
    try:
        FM.parse("---\nname: x\ndescription: y\n")
        no_close = None
    except FM.FrontmatterError as e:
        no_close = str(e)
    return {"rows": rows, "no_opening": no_open, "no_closing": no_close}


def e3_sizes(skills):
    sz = B.sizes(skills)
    main = next(s for s in skills if s["name"] == "weekly-inquiry-report")
    return {"project_agents_md": B.project_tokens(), "skills": sz,
            "sum_meta": sum(x["meta"] for x in sz), "sum_skill_md": sum(x["skill_md"] for x in sz),
            "sum_refs": sum(x["refs_total"] for x in sz),
            "main_meta_text": B.meta_text(main),
            "main_meta_chars": len(B.meta_text(main)),
            "main_meta_ascii": sum(1 for ch in B.meta_text(main) if ord(ch) < 128)}


def e4_week(skills):
    res = B.week(skills)
    by_rule = []
    for rule in B.load_rules():
        r = B.week(skills, rule)
        by_rule.append({"rule": rule["label"], "paste_all": r["paste_all_total"], "staged": r["staged_total"],
                        "saving_pct": f(100 * r["saving"], 1)})
    rows = [{"id": r["id"], "task": r["task"], "skill": r["skill"], "paste_all": r["paste_all"],
             "staged": r["staged"], **{f"part_{k}": v for k, v in r["parts"].items()}} for r in res["rows"]]
    return {"conversations": len(rows), "rows": rows, "paste_all_total": res["paste_all_total"],
            "staged_total": res["staged_total"], "staged_ratio": f(res["staged_ratio"]),
            "saving_pct": f(100 * res["saving"], 1), "by_rule": by_rule}


def e5_scale(skills):
    sz = B.sizes(skills)
    m = F(sum(x["meta"] for x in sz), len(sz))
    b = F(sum(x["body"] + x["refs_total"] for x in sz), len(sz))
    ns = [1, 5, 10, 20, 50, 100]
    ours = [{"n": r["n"], "paste_all": f(r["paste_all"], 1), "staged": f(r["staged"], 1),
             "meta_only": f(r["meta_only"], 1), "ratio": f(r["ratio"]), "saving_pct": f(100 * (1 - r["ratio"]), 1)}
            for r in B.scale(m, b, ns)]
    doc = [{"n": r["n"], "paste_all": r["paste_all"], "staged": r["staged"], "meta_only": r["meta_only"],
            "ratio": f(r["ratio"]), "saving_pct": f(100 * (1 - r["ratio"]), 1)}
           for r in B.scale(100, 5000, ns)]
    return {"ours_m": f(m, 2), "ours_b": f(b, 2), "ours": ours, "ours_limit_ratio": f(m / (m + b)),
            "doc_m": 100, "doc_b": 5000, "doc": doc, "doc_limit_ratio": f(F(100, 5100))}


def e6_round_trip(skills):
    sz = B.sizes(skills)
    w = B.load_week()
    rt = w["round_trip"]
    out = []
    for sc in rt["scenarios"]:
        conv = w["conversations"][sc["conversation"] - 1]
        for batch in (False, True):
            r = B.round_trip(sz, B.project_tokens(), conv, sc["user_tokens"], rt["tool_call_tokens"],
                             rt["w"], rt["r"], batch=batch)
            out.append({"scenario": sc["id"], "label": sc["label"], "user_tokens": sc["user_tokens"], "batch": batch,
                        "paste_all_calls": len(r["paste_all_inputs"]), "staged_calls": len(r["staged_inputs"]),
                        "paste_all_inputs": r["paste_all_inputs"], "staged_inputs": r["staged_inputs"],
                        "paste_all_nocache": r["paste_all_nocache"], "paste_all_cached": f(r["paste_all_cached"], 2),
                        "staged_nocache": r["staged_nocache"], "staged_cached": f(r["staged_cached"], 2),
                        "staged_vs_paste_nocache": f(F(r["staged_nocache"], r["paste_all_nocache"]), 3),
                        "staged_vs_paste_cached": f(r["staged_cached"] / r["paste_all_cached"], 3),
                        "staged_cached_vs_paste_nocache": f(r["staged_cached"] / r["paste_all_nocache"], 3)})
    return {"w": rt["w"], "r": rt["r"], "tool_call_tokens": rt["tool_call_tokens"], "rows": out}


def e7_one_file(skills):
    vend = json.loads(B.VENDOR_FILE.read_text(encoding="utf-8"))
    text = B.all_in_one_file_text(skills)
    proj = B.PROJECT_FILE.read_text(encoding="utf-8")
    return {"all_in_one_bytes": B.utf8_bytes(text), "all_in_one_lines": len(text.split("\n")),
            "all_in_one_chars": len(text),
            "codex_limit_bytes": vend["codex"]["project_doc_max_bytes"],
            "claude_md_target_lines": vend["claude_code"]["claude_md_target_lines"],
            "agents_md_bytes": B.utf8_bytes(proj), "agents_md_lines": len(proj.strip().split("\n")),
            "bytes_per_hangul": len("가".encode("utf-8")),
            "skills_to_fill_codex_limit": f(F(vend["codex"]["project_doc_max_bytes"], B.utf8_bytes(text)) * len(skills), 1)}


def e8_crosscheck(n_cases=300):
    rng = random.Random(SEED)
    bad = 0
    for _ in range(n_cases):
        n = rng.randint(1, 60)
        ms = [rng.randint(30, 200) for _ in range(n)]
        bs = [rng.randint(100, 6000) for _ in range(n)]
        used = rng.randrange(n)
        loop_all = sum(mi + bi for mi, bi in zip(ms, bs))
        loop_st = sum(ms) + bs[used]
        m_avg, b_avg = F(sum(ms), n), F(sum(bs), n)
        closed_all = n * (m_avg + b_avg)
        if loop_all != closed_all:
            bad += 1
        if loop_st != sum(ms) + bs[used]:
            bad += 1
        # 글자당 토큰이 상수 c 하나뿐이고 올림을 하지 않으면, 비율은 c와 상관없다
        chars_m = [mi * 4 for mi in ms]
        chars_b = [bi * 4 for bi in bs]
        ratios = set()
        for c in (F(1, 4), F(1, 2), F(1), F(2)):
            st = c * sum(chars_m) + c * chars_b[used]
            al = c * (sum(chars_m) + sum(chars_b))
            ratios.add(st / al)
        if len(ratios) != 1:
            bad += 1
    return {"cases": n_cases, "mismatches": bad}


def main():
    skills = FM.load_skills()
    vend = json.loads(B.VENDOR_FILE.read_text(encoding="utf-8"))
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "seed": SEED, "vendor_file": B.VENDOR_FILE.name, "checked": vend["checked"],
                 "token_rule": B.DEFAULT_RULE},
        "e0_hand": e0_hand(),
        "e1_parse": e1_parse(skills),
        "e2_bad": e2_bad(),
        "e3_sizes": e3_sizes(skills),
        "e4_week": e4_week(skills),
        "e5_scale": e5_scale(skills),
        "e6_round_trip": e6_round_trip(skills),
        "e7_one_file": e7_one_file(skills),
        "e8_crosscheck": e8_crosscheck(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
