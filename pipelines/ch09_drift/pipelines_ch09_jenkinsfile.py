"""9장 Jenkinsfile 검사. stages()는 7장 pipelines_ch07_jenkinsfile.py(6장에서 복사한 것)에서 복사했다.

규칙
1. 예약 실행(cron) 트리거가 있고, 분 자리에 H(작업 이름 해시)를 쓴다. 모든 작업이 정각에 몰리지 않게.
2. Jenkins 단계 어디에서도 학습을 직접 돌리지 않는다(학습은 KFP run에서만).
3. 경보가 났을 때 재학습 run을 제출하는 단계가 있고, 드리프트 검사보다 뒤에 있다.
4. 되돌리기 단계는 판정(Gate) 뒤에 있고, 이미지를 다이제스트(@)로 가리킨다. ':latest' 같은 태그로 되돌리면 실패.
"""

import re

TRAIN_PATTERNS = [r"fit_logistic", r"\btrain\.py\b", r"steps\.py\s+train", r"M\.train\("]


def stages(text):
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
    cron = re.search(r"cron\('([^']+)'\)", text)
    if not cron:
        problems.append("예약 실행(cron) 트리거가 없다")
    elif not cron.group(1).split()[0].startswith("H"):
        problems.append(f"cron 분 자리에 H를 쓰지 않았다: {cron.group(1)}")
    st = stages(text)
    names = [n for n, _ in st]
    for n, b in st:
        for pat in TRAIN_PATTERNS:
            if re.search(pat, b):
                problems.append(f"'{n}' 단계가 Jenkins에서 학습을 직접 돌린다({pat})")
    drift = next((n for n in names if "drift" in n.lower()), None)
    submit = next((n for n in names if "retrain" in n.lower()), None)
    gate = next((n for n in names if n.lower().startswith("gate")), None)
    rollback = next((n for n in names if "rollback" in n.lower()), None)
    if drift is None:
        problems.append("드리프트 검사 단계가 없다")
    if submit is None:
        problems.append("재학습 run을 제출하는 단계가 없다")
    elif drift and names.index(submit) < names.index(drift):
        problems.append("검사 전에 재학습을 제출한다")
    if rollback is None:
        problems.append("되돌리기 단계가 없다")
    else:
        body = dict(st)[rollback]
        if gate is None or names.index(rollback) < names.index(gate):
            problems.append("되돌리기가 판정보다 먼저 온다")
        if "@" not in body:
            problems.append("되돌리기가 다이제스트가 아니라 태그를 가리킨다")
        if re.search(r":latest\b", body):
            problems.append("되돌리기에 ':latest'를 쓴다")
    return problems
