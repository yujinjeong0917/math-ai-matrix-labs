"""생산성 도구 4장: 일반 페이지 방식(관계 없음)과 찾는 사람의 규칙.

규칙은 specs/question_rules.md에 실험 전에 적어 두었다. 찾는 사람은 페이지에 글자로 적힌 것만 믿는다.
"""

import re

import productivity_ch04_data as DATA

DONE = DATA.DONE


def build_pages(lost=frozenset()):
    """정답 데이터에서 페이지 묶음을 만든다. lost에 든 연결은 글로 적히지 않는다."""
    decision_pages = {}
    for d, v in sorted(DATA.DECISIONS.items()):
        lines = [f"# {d} {v['title']}", f"날짜: {v['date']}"]
        if ("owner", d, v["owner"]) not in lost:
            lines.append(f"결정자: {v['owner']}")
        lines.append("내용: 회의에서 정한 내용을 적었어요.")
        for t in sorted(DATA.TASKS):
            if d in DATA.TASKS[t]["decisions"] and ("task", d, t) not in lost:
                lines.append(f"할 일: {t} {DATA.TASKS[t]['title']}")
        for s in v["specs"]:
            if ("spec", d, s) not in lost:
                lines.append(f"화면: {s}")
        if d in DATA.NO_SPEC_OK:
            lines.append("화면: 없음(일정 결정)")
        if v["supersedes"]:
            lines.append(f"번복: {v['supersedes']}")
        decision_pages[d] = lines
    task_list = ["# 할 일 목록"]
    for t, v in sorted(DATA.TASKS.items()):
        box = "[x]" if v["status"] == DONE else "[ ]"
        task_list.append(f"- {box} {t} {v['title']} @{v['owner']}")
    # 할 일 목록은 페이지 하나라 마지막 수정 시각이 하나뿐이다.
    task_list_edited = max(v["updated"] for v in DATA.TASKS.values())
    spec_pages = {s: {"lines": [f"# {s} {v['title']}", "문구와 동작을 적었어요."], "edited": v["edited"]}
                  for s, v in DATA.SPECS.items()}
    return {"decisions": decision_pages, "task_list": task_list, "task_list_edited": task_list_edited, "specs": spec_pages}


class Reader:
    """찾는 사람. 읽은 줄 수를 센다."""

    def __init__(self, pages):
        self.p = pages
        self.lines_read = 0

    def read(self, lines):
        self.lines_read += len(lines)
        return lines

    def tasks_in(self, d):
        return [m.group(1) for ln in self.read(self.p["decisions"][d]) if (m := re.match(r"할 일: (T\d+)", ln))]

    def task_status(self):
        out = {}
        for ln in self.read(self.p["task_list"]):
            m = re.match(r"- \[(x| )\] (T\d+)", ln)
            if m:
                out[m.group(2)] = m.group(1) == "x"
        return out


UNANSWERABLE = "답할 수 없음"


def q1(pages, keyword="취소 규정"):
    r = Reader(pages)
    hit = [d for d, lines in pages["decisions"].items() if keyword in r.read(lines[:1])[0]]  # 제목만 훑는다
    tasks = sorted({t for d in hit for t in r.tasks_in(d)})
    return (tasks if tasks else UNANSWERABLE), r.lines_read


def q2(pages, s="S03"):
    r = Reader(pages)
    hit = sorted(d for d, lines in pages["decisions"].items() if f"화면: {s}" in r.read(lines))
    return hit, r.lines_read


def q3(pages):
    r = Reader(pages)
    done = r.task_status()
    out = []
    for d in sorted(pages["decisions"]):
        if any(not done[t] for t in r.tasks_in(d)):
            out.append(d)
    return out, r.lines_read


def q4(pages):
    r = Reader(pages)
    out = {}
    for d, lines in sorted(pages["decisions"].items()):
        owner = None
        for ln in r.read(lines):
            if ln.startswith("결정자: "):
                owner = ln.split(": ", 1)[1]
        out[d] = owner
    if any(v is None for v in out.values()):
        return UNANSWERABLE, r.lines_read, out
    return out, r.lines_read, out


def q5(pages):
    r = Reader(pages)
    r.read(pages["task_list"])
    for s in pages["specs"].values():
        r.read(s["lines"])
    # 할 일마다 수정 시각이 없어 스펙 수정일과 비교할 수 없다.
    return UNANSWERABLE, r.lines_read


def answer_all(pages):
    a1, n1 = q1(pages)
    a2, n2 = q2(pages)
    a3, n3 = q3(pages)
    a4, n4, _ = q4(pages)
    a5, n5 = q5(pages)
    return {"Q1": a1, "Q2": a2, "Q3": a3, "Q4": a4, "Q5": a5}, {"Q1": n1, "Q2": n2, "Q3": n3, "Q4": n4, "Q5": n5}


def total_lines(pages):
    return sum(len(v) for v in pages["decisions"].values()) + len(pages["task_list"]) + sum(
        len(s["lines"]) for s in pages["specs"].values())
