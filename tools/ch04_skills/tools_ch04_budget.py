"""스킬을 "전부 싣기"와 "앞머리만 + 쓰는 것만 본문"으로 실을 때 드는 토큰 수.

토큰은 실제 토크나이저로 세지 않는다. data/token_rule.json에 정한 글자 수 근사 규칙(가정값)을 쓴다.
금액·배율 계산은 fractions로 정확히 한다. 표준 라이브러리만 쓴다.
"""

import json
import math
from fractions import Fraction as F
from pathlib import Path

import tools_ch04_frontmatter as FM

HERE = Path(__file__).parent
DATA = HERE / "data"
RULE_FILE = DATA / "token_rule.json"
WEEK_FILE = DATA / "week_tasks.json"
PROJECT_FILE = DATA / "project" / "AGENTS.md"
VENDOR_FILE = DATA / "vendors_2026-10-10.json"


def load_rules():
    r = json.loads(RULE_FILE.read_text(encoding="utf-8"))
    return [r["default"]] + r["alternatives"]


DEFAULT_RULE = load_rules()[0]


def tokens(text, rule=None):
    """글 -> 근사 토큰 수. ASCII 글자는 rule 글자당 1토큰, 그 밖의 글자는 글자당 rule 토큰(각각 올림)."""
    rule = rule or DEFAULT_RULE
    ascii_n = sum(1 for ch in text if ord(ch) < 128)
    other_n = len(text) - ascii_n
    a = math.ceil(F(ascii_n) / F(str(rule["ascii_chars_per_token"])))
    b = math.ceil(F(other_n) * F(str(rule["non_ascii_tokens_per_char"])))
    return a + b


def meta_text(skill):
    """모델이 늘 보는 목록 한 줄: 이름과 설명."""
    return f"{skill['fm']['name']}: {skill['fm']['description']}"


def is_reference(path):
    """읽으면 문맥에 들어가는 딸린 파일. scripts/ 아래 코드는 실행 결과만 들어가므로 뺀다."""
    return not path.startswith("scripts/")


def sizes(skills, rule=None):
    """스킬마다 앞머리(목록 한 줄)·본문·딸린 파일 토큰 수."""
    out = []
    for s in skills:
        refs = {p: tokens(t, rule) for p, t in s["files"].items() if is_reference(p)}
        out.append({"name": s["name"], "meta": tokens(meta_text(s), rule), "body": tokens(s["body"], rule),
                    "skill_md": tokens(s["skill_md"], rule), "refs": refs, "refs_total": sum(refs.values()),
                    "scripts": [p for p in s["files"] if not is_reference(p)]})
    return out


def project_tokens(rule=None):
    return tokens(PROJECT_FILE.read_text(encoding="utf-8"), rule)


def paste_all_per_conversation(sz, project):
    """전부 싣기: 프로젝트 지침 + 모든 SKILL.md(앞머리 포함) + 모든 딸린 파일을 대화마다."""
    return project + sum(x["skill_md"] + x["refs_total"] for x in sz)


def staged_per_conversation(sz, project, conv, rule=None):
    """단계적 공개: 프로젝트 지침 + 모든 스킬의 목록 한 줄 + (쓴 스킬의 본문 + 읽은 파일 + 스크립트 출력)."""
    total = project + sum(x["meta"] for x in sz)
    parts = {"project": project, "meta": sum(x["meta"] for x in sz), "body": 0, "files": 0, "script_output": 0}
    if conv.get("skill"):
        x = next(s for s in sz if s["name"] == conv["skill"])
        parts["body"] = x["body"]
        parts["files"] = sum(x["refs"][p] for p in conv.get("files", []))
        if conv.get("script_output"):
            parts["script_output"] = tokens(conv["script_output"], rule)
    total = sum(parts.values())
    return total, parts


def load_week():
    return json.loads(WEEK_FILE.read_text(encoding="utf-8"))


def week(skills=None, rule=None):
    skills = skills or FM.load_skills()
    sz = sizes(skills, rule)
    proj = project_tokens(rule)
    w = load_week()
    rows = []
    for c in w["conversations"]:
        st, parts = staged_per_conversation(sz, proj, c, rule)
        rows.append({"id": c["id"], "task": c["task"], "skill": c["skill"],
                     "paste_all": paste_all_per_conversation(sz, proj), "staged": st, "parts": parts})
    pa = sum(r["paste_all"] for r in rows)
    sg = sum(r["staged"] for r in rows)
    return {"rows": rows, "paste_all_total": pa, "staged_total": sg,
            "staged_ratio": F(sg, pa), "saving": 1 - F(sg, pa)}


def scale(m, b, n_list, k=1):
    """스킬 N개 중 k개를 쓸 때 대화 하나의 토큰: 전부 N(m+b), 단계적 Nm + kb. (m: 목록 한 줄, b: 본문과 딸린 파일)"""
    rows = []
    for n in n_list:
        allt = n * (m + b)
        st = n * m + k * b
        rows.append({"n": n, "paste_all": allt, "staged": st, "meta_only": n * m,
                     "ratio": F(st, allt)})
    return rows


def _bill(inputs, w, r):
    """호출마다의 입력 목록 -> (캐싱 없음, 캐싱) 기본가 토큰어치.

    캐싱(쓰기 w, 읽기 r): 첫 호출은 전부 쓰기, 다음 호출부터 앞 호출까지의 입력은 읽기로,
    새로 붙은 부분(도구 호출 + 읽은 내용)은 쓰기로 낸다.
    """
    nocache = sum(inputs)
    cached = F(str(w)) * inputs[0]
    for prev, cur in zip(inputs, inputs[1:]):
        cached += F(str(r)) * prev + F(str(w)) * (cur - prev)
    return nocache, cached


def round_trip(sz, project, conv, user_tokens, tool_call, w, r, rule=None, batch=False):
    """한 대화를 모델 호출 단위로 센다(입력 토큰을 기본가 몇 토큰어치로 내는지).

    전부 싣기: 첫 호출 입력 = 프로젝트 + 전부 + 사용자 메시지. 스크립트를 돌려야 하면 호출 1번 더.
    단계적 공개: 첫 호출(목록만 보고 스킬을 고름) + 읽는 것 하나마다 호출 1번(batch=False).
      batch=True면 본문을 읽는 호출 1번, 딸린 파일과 스크립트를 한꺼번에 읽는 호출 1번으로 묶는다.
    """
    x = next(s for s in sz if s["name"] == conv["skill"])
    script = tokens(conv["script_output"], rule) if conv.get("script_output") else None
    pa = [paste_all_per_conversation(sz, project) + user_tokens]
    if script is not None:
        pa.append(pa[-1] + tool_call + script)
    refs = [x["refs"][p] for p in conv.get("files", [])]
    after_body = refs + ([script] if script is not None else [])
    if batch:
        steps = [(tool_call, x["body"])]
        if after_body:
            steps.append((tool_call * len(after_body), sum(after_body)))
    else:
        steps = [(tool_call, item) for item in [x["body"]] + after_body]
    st = [project + sum(s["meta"] for s in sz) + user_tokens]
    for call, item in steps:
        st.append(st[-1] + call + item)
    pa_no, pa_ca = _bill(pa, w, r)
    st_no, st_ca = _bill(st, w, r)
    return {"paste_all_inputs": pa, "staged_inputs": st,
            "paste_all_nocache": pa_no, "paste_all_cached": pa_ca,
            "staged_nocache": st_no, "staged_cached": st_ca}


def utf8_bytes(text):
    return len(text.encode("utf-8"))


def all_in_one_file_text(skills):
    """모든 스킬 내용을 프로젝트 지침 파일 하나에 몰아 넣었을 때의 글."""
    parts = [PROJECT_FILE.read_text(encoding="utf-8")]
    for s in skills:
        parts.append(s["skill_md"])
        parts.extend(t for p, t in s["files"].items() if is_reference(p))
    return "\n\n".join(parts)
