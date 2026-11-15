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
    """토큰 근사 규칙 목록: [기본, 대안...]. 대안은 규칙 선택에 결과가 얼마나 민감한지 보려는 것이다."""
    r = json.loads(RULE_FILE.read_text(encoding="utf-8"))
    return [r["default"]] + r["alternatives"]


DEFAULT_RULE = load_rules()[0]


def tokens(text, rule=None):
    """글 -> 근사 토큰 수. ASCII 글자는 rule 글자당 1토큰, 그 밖의 글자는 글자당 rule 토큰(각각 올림)."""
    rule = rule or DEFAULT_RULE
    ascii_n = sum(1 for ch in text if ord(ch) < 128)
    other_n = len(text) - ascii_n
    # ASCII: 글자 수 / (토큰당 글자 수), 올림. Fraction이라 부동소수점 오차 없이 올림한다.
    a = math.ceil(F(ascii_n) / F(str(rule["ascii_chars_per_token"])))
    # 한글 등: 글자 수 x (글자당 토큰), 올림. 실제 토크나이저와 다른 근사라 절대값보다 비율을 읽는다.
    b = math.ceil(F(other_n) * F(str(rule["non_ascii_tokens_per_char"])))
    return a + b


def meta_text(skill):
    """모델이 늘 보는 목록 한 줄: 이름과 설명."""
    # 목록 한 줄의 형식은 frontmatter.listing_line과 같다(여기서는 자르지 않는다).
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
    """프로젝트 지침 파일(AGENTS.md)의 근사 토큰 수. 모든 대화에 항상 실린다."""
    return tokens(PROJECT_FILE.read_text(encoding="utf-8"), rule)


def paste_all_per_conversation(sz, project):
    """전부 싣기: 프로젝트 지침 + 모든 SKILL.md(앞머리 포함) + 모든 딸린 파일을 대화마다."""
    return project + sum(x["skill_md"] + x["refs_total"] for x in sz)


def staged_per_conversation(sz, project, conv, rule=None):
    """단계적 공개: 프로젝트 지침 + 모든 스킬의 목록 한 줄 + (쓴 스킬의 본문 + 읽은 파일 + 스크립트 출력)."""
    # 아래 parts 합으로 다시 구하므로 이 값은 곧 덮어쓰인다(사용되지 않는 중간값).
    total = project + sum(x["meta"] for x in sz)
    parts = {"project": project, "meta": sum(x["meta"] for x in sz), "body": 0, "files": 0, "script_output": 0}
    if conv.get("skill"):
        # 이 대화가 쓰는 스킬. skill이 없는 대화는 이 블록을 건너뛰어 body, files, script_output이 0으로 남는다.
        x = next(s for s in sz if s["name"] == conv["skill"])
        parts["body"] = x["body"]
        parts["files"] = sum(x["refs"][p] for p in conv.get("files", []))
        if conv.get("script_output"):
            parts["script_output"] = tokens(conv["script_output"], rule)
    total = sum(parts.values())
    return total, parts


def load_week():
    """한 주 대화 15개와 왕복 시나리오 가정값(data/week_tasks.json)을 읽는다."""
    return json.loads(WEEK_FILE.read_text(encoding="utf-8"))


def week(skills=None, rule=None):
    """한 주(대화 15번)의 지시문 토큰 합계: 전부 싣기 대 단계적 공개. 절감률 = 1 - 단계적 / 전부."""
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
        # 전부 싣기: 스킬 N개의 (목록 한 줄 + 본문)을 모두. 단계적 공개는 N개의 목록 한 줄 + 쓰는 k개의 본문.
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
    # 첫 호출은 전부 새로 쓰므로 w배. 단위는 '기본 입력가 몇 토큰어치'.
    cached = F(str(w)) * inputs[0]
    for prev, cur in zip(inputs, inputs[1:]):
        # 다음 호출: 앞 호출까지의 입력은 캐시 읽기(r배), 새로 늘어난 부분만 쓰기(w배). 매 호출이 직전 호출의 입력을 그대로 앞부분으로 가진다는 가정(모형의 단순화).
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
    # 각 원소는 호출 하나의 입력 토큰 수. 전부 싣기는 첫 호출에 모든 것이 들어 있다.
    pa = [paste_all_per_conversation(sz, project) + user_tokens]
    if script is not None:
        pa.append(pa[-1] + tool_call + script)
    refs = [x["refs"][p] for p in conv.get("files", [])]
    # 본문을 읽은 뒤 더 읽을 것: 딸린 파일과 스크립트 출력. 스크립트 코드 자체는 읽지 않고 실행 결과만 문맥에 들어간다.
    after_body = refs + ([script] if script is not None else [])
    if batch:
        # batch: 본문을 읽는 호출 1번 + 나머지를 한꺼번에 읽는 호출 1번. 호출이 늘면 앞 입력이 반복해서 청구되므로 호출 수가 비용을 좌우한다.
        steps = [(tool_call, x["body"])]
        if after_body:
            steps.append((tool_call * len(after_body), sum(after_body)))
    else:
        steps = [(tool_call, item) for item in [x["body"]] + after_body]
    # 단계적 공개의 첫 호출: 프로젝트 지침 + 모든 스킬의 목록 한 줄 + 사용자 메시지.
    st = [project + sum(s["meta"] for s in sz) + user_tokens]
    for call, item in steps:
        st.append(st[-1] + call + item)
    pa_no, pa_ca = _bill(pa, w, r)
    st_no, st_ca = _bill(st, w, r)
    return {"paste_all_inputs": pa, "staged_inputs": st,
            "paste_all_nocache": pa_no, "paste_all_cached": pa_ca,
            "staged_nocache": st_no, "staged_cached": st_ca}


def utf8_bytes(text):
    """UTF-8 바이트 수. 한글은 글자당 3바이트라 글자 수보다 크다(Codex 한도는 바이트 기준)."""
    return len(text.encode("utf-8"))


def all_in_one_file_text(skills):
    """모든 스킬 내용을 프로젝트 지침 파일 하나에 몰아 넣었을 때의 글."""
    parts = [PROJECT_FILE.read_text(encoding="utf-8")]
    for s in skills:
        parts.append(s["skill_md"])
        parts.extend(t for p, t in s["files"].items() if is_reference(p))
    return "\n\n".join(parts)
