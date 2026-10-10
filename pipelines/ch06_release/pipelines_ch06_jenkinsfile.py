"""6장 Jenkinsfile 검사: 서버 없이 단계 순서와 승인 규칙만 읽는다.

5장 러너(pipelines_ch05_jenkins.py)처럼 실행하지는 않는다. 이 장에서 볼 것은 '무엇이 무엇보다 먼저 오나'다.
규칙
1. 단계 순서: 오프라인 검사 -> 카나리 -> 사람 승인 -> 100% 전환.
2. 승인 단계에는 input이 있고 submitter로 누를 수 있는 사람을 좁혔다(문서 기본값은 '누구나').
3. 카나리 단계가 실패하면 post { failure }에서 v2 비율을 0으로 되돌린다.
4. 100%로 바꾸는 명령(--v2 1.0)은 승인 단계 뒤에만 있다.
"""

import re


def stages(text):
    """(이름, 본문) 목록. 중괄호 짝을 세어 stage('...') { ... } 블록을 자른다."""
    out = []
    for m in re.finditer(r"stage\('([^']+)'\)\s*\{", text):
        depth, k = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append((m.group(1), text[m.end():k - 1]))
    return out


def lint(text):
    problems = []
    st = stages(text)
    names = [n for n, _ in st]
    body = dict(st)
    canary = next((n for n in names if "Canary" in n), None)
    approve = next((n for n, b in st if re.search(r"\binput\s*\{", b)), None)
    full = [n for n, b in st if "--v2 1.0" in b]
    if canary is None:
        problems.append("카나리 단계가 없다")
    if approve is None:
        problems.append("사람 승인(input) 단계가 없다")
    elif "submitter" not in body[approve]:
        problems.append("승인 단계에 submitter가 없어 누구나 누를 수 있다")
    if canary and approve and names.index(canary) > names.index(approve):
        problems.append("승인이 카나리보다 먼저 온다")
    for n in full:
        if approve is None or names.index(n) < names.index(approve):
            problems.append(f"'{n}' 단계가 승인 전에 100%로 바꾼다")
    if canary and not re.search(r"failure\s*\{[^}]*--v2 0\.0", body[canary]):
        problems.append("카나리가 실패해도 v2 비율을 되돌리지 않는다")
    return problems
