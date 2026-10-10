"""7장 Jenkinsfile 검사: Jenkins와 KFP의 경계를 규칙으로 읽는다. stages()는 6장 pipelines_ch06_jenkinsfile.py에서 복사했다.

규칙
1. 'Submit KFP run' 단계가 있고, 이미지를 올리는 단계보다 뒤에 있다.
2. 제출에 쓰는 이미지 이름이 커밋 SHA(GIT_COMMIT)나 다이제스트(@sha256:)로 고정돼 있다. ':latest'는 실패.
3. Jenkins 단계 어디에서도 학습 명령을 직접 돌리지 않는다(학습은 KFP train task에서만).
4. 판정(Gate) 단계는 제출 뒤에 있고, KFP run에서 받아 온 지표 파일을 읽는다.
"""

import re

TRAIN_PATTERNS = [r"churn\.train", r"steps\.py\s+train", r"\btrain\.py\b", r"fit_logistic"]


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
    st = stages(text)
    names = [n for n, _ in st]
    body = dict(st)
    submit = next((n for n in names if "KFP" in n), None)
    push = next((n for n, b in st if "docker push" in b), None)
    gate = next((n for n in names if n.lower().startswith("gate")), None)
    if submit is None:
        problems.append("KFP run을 제출하는 단계가 없다")
    if push is None:
        problems.append("이미지를 올리는 단계가 없다")
    if submit and push and names.index(submit) < names.index(push):
        problems.append("이미지를 올리기 전에 run을 제출한다")
    image = re.search(r"IMAGE\s*=\s*[\"']([^\"']+)[\"']", text)
    ref = image.group(1) if image else ""
    if not ref:
        problems.append("제출할 이미지 이름(IMAGE)이 없다")
    elif ref.endswith(":latest") or ":" not in ref.split("/")[-1] and "@" not in ref:
        problems.append(f"이미지가 움직이는 태그다: {ref}")
    elif "GIT_COMMIT" not in ref and "@sha256:" not in ref:
        problems.append(f"이미지가 커밋 SHA나 다이제스트로 고정돼 있지 않다: {ref}")
    for n, b in st:
        for pat in TRAIN_PATTERNS:
            if re.search(pat, b):
                problems.append(f"'{n}' 단계가 Jenkins에서 학습을 직접 돌린다({pat})")
    if gate is None:
        problems.append("판정(Gate) 단계가 없다")
    elif submit and names.index(gate) < names.index(submit):
        problems.append("판정이 run 제출보다 먼저 온다")
    elif "kfp_metrics" not in body[gate]:
        problems.append("판정이 KFP run의 지표를 읽지 않는다")
    return problems
