"""Jenkinsfile(Declarative Pipeline)의 작은 부분집합을 읽고 실제로 실행하는 흉내 러너.

Jenkins 서버를 띄우지 않는다. 대신 Jenkins 공식 문서에 적힌 동작을 아래 규칙으로 옮겼다.
- stages는 위에서부터 차례로 돈다. 한 단계의 sh가 0이 아닌 종료 코드를 내면 그 단계와 빌드가 FAILURE가 되고,
  남은 단계는 건너뛴다("otherwise the Pipeline would have exited early", jenkins.io Using a Jenkinsfile).
- junit 단계는 JUnit XML에서 실패를 하나라도 찾으면 빌드를 UNSTABLE로 표시한다(JUnit 플러그인 문서).
  UNSTABLE이어도 다음 단계는 계속 돈다. options { skipStagesAfterUnstable() }가 있으면 건너뛴다(Pipeline Syntax).
- when { expression { ... } }는 currentBuild.result를 쓰는 간단한 비교만 지원한다.
- archiveArtifacts가 파일을 하나도 못 찾으면 빌드가 FAILURE가 된다(allowEmptyArchive 기본값, Pipeline Basic Steps 문서).
- post의 always / success / unstable / failure 블록은 빌드 결과에 따라 마지막에 돈다.
지원하는 단계: checkout scm(아무것도 안 함), sh '...', junit '...', archiveArtifacts '...'.
그 밖의 문법을 만나면 조용히 넘기지 않고 오류를 낸다.
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

TOKEN = re.compile(r"\s*(?:(//[^\n]*)|('(?:[^'\\]|\\.)*')|(\"(?:[^\"\\]|\\.)*\")|([{}()])|([A-Za-z_][\w./]*)|(\|\||&&|==|!=)|(\S))")


def tokenize(text):
    toks, pos = [], 0
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m or m.end() == pos:
            break
        pos = m.end()
        com, s1, s2, br, ident, op, other = m.groups()
        if com:
            continue
        if s1 or s2:
            toks.append(("str", (s1 or s2)[1:-1]))
        elif br:
            toks.append(("br", br))
        elif ident:
            toks.append(("id", ident))
        elif op:
            toks.append(("op", op))
        elif other and other.strip():
            toks.append(("other", other))
    return toks


def _parse_block(toks, i):
    """'{' 다음부터 '}'까지 문장 목록을 읽는다. 문장 = (이름, 인자 목록, 하위 블록 또는 None, 원문 토큰)."""
    items = []
    while i < len(toks):
        kind, val = toks[i]
        if (kind, val) == ("br", "}"):
            return items, i + 1
        if kind != "id":
            raise SyntaxError(f"이름이 와야 하는 자리에 {val!r}")
        name, i = val, i + 1
        args = []
        if i + 1 < len(toks) and toks[i] == ("other", "=") and toks[i + 1][0] == "str":  # environment의 NAME = '값'
            args.append(toks[i + 1][1])
            i += 2
        elif i < len(toks) and toks[i] == ("br", "("):
            depth, i = 1, i + 1
            while depth:
                k, v = toks[i]
                if (k, v) == ("br", "("):
                    depth += 1
                elif (k, v) == ("br", ")"):
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                if depth:
                    args.append(v)
                i += 1
        elif i < len(toks) and toks[i][0] == "str":
            args.append(toks[i][1])
            i += 1
        elif i < len(toks) and toks[i][0] == "id" and name in ("checkout", "agent"):
            args.append(toks[i][1])
            i += 1
        body = None
        if i < len(toks) and toks[i] == ("br", "{"):
            if name == "expression":
                depth, j, raw = 1, i + 1, []
                while depth:
                    k, v = toks[j]
                    depth += (k, v) == ("br", "{")
                    depth -= (k, v) == ("br", "}")
                    if depth:
                        raw.append((k, v))
                    j += 1
                body, i = raw, j
            else:
                body, i = _parse_block(toks, i + 1)
        items.append((name, args, body))
    return items, i


def parse(text):
    toks = tokenize(text)
    if toks[:2] != [("id", "pipeline"), ("br", "{")]:
        raise SyntaxError("pipeline { ... } 로 시작해야 한다")
    items, _ = _parse_block(toks, 2)
    top = {name: (args, body) for name, args, body in items}
    stages = []
    for name, args, body in top["stages"][1]:
        if name != "stage":
            raise SyntaxError(f"stages 안에는 stage만: {name}")
        parts = {n: b for n, a, b in body}
        stages.append({"name": args[0], "steps": parts.get("steps", []), "when": parts.get("when")})
    options = [n for n, a, b in (top.get("options", ([], []))[1] or [])]
    env = {n: a[0] for n, a, b in (top.get("environment", ([], []))[1] or [])}
    post = {n: b for n, a, b in (top.get("post", ([], []))[1] or [])}
    return {"stages": stages, "options": options, "environment": env, "post": post}


def eval_when(raw, result):
    """currentBuild.result 비교식만 계산한다. result가 None이면 아직 문제없이 도는 중(SUCCESS와 같음)."""
    words = []
    for k, v in raw:
        if v == "currentBuild.result":
            words.append(repr(result))
        elif v == "null":
            words.append("None")
        elif v == "||":
            words.append("or")
        elif v == "&&":
            words.append("and")
        elif k == "str":
            words.append(repr(v))
        elif v in ("==", "!=", "(", ")"):
            words.append(v)
        else:
            raise SyntaxError(f"when expression에서 지원하지 않는 토큰: {v}")
    return bool(eval(" ".join(words), {"__builtins__": {}}, {}))


def read_junit(paths):
    cases = []
    for p in paths:
        root = ET.parse(p).getroot()
        for tc in root.iter("testcase"):
            bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
            name = f"{tc.get('classname')}::{tc.get('name')}"
            msg = None if bad is None else (bad.get("message") or "").splitlines()[0][:160]
            cases.append({"name": name, "ok": bad is None and tc.find("skipped") is None, "message": msg})
    return cases


WORSE = {None: 0, "SUCCESS": 0, "UNSTABLE": 1, "FAILURE": 2}


def run_pipeline(text, workspace, extra_env=None, archive_dir=None):
    """workspace(이미 체크아웃된 폴더)에서 파이프라인을 돌린다. 단계별 상태·시간, 테스트 결과, 빌드 결과를 돌려준다."""
    p = parse(text)
    ws = Path(workspace)
    env = {**os.environ, **p["environment"], **(extra_env or {})}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    state = {"result": None, "tests": [], "archived": [], "log": []}

    def mark(r):
        if WORSE[r] > WORSE[state["result"]]:
            state["result"] = r

    def do_step(name, args):
        if name == "checkout":
            return "SUCCESS"
        if name == "sh":
            r = subprocess.run(["bash", "-c", args[0]], cwd=ws, env=env, capture_output=True, text=True)
            state["log"].append({"cmd": args[0], "code": r.returncode, "tail": (r.stdout + r.stderr).strip().splitlines()[-3:]})
            if r.returncode != 0:
                mark("FAILURE")
                return "FAILURE"
            return "SUCCESS"
        if name == "junit":
            files = sorted(glob.glob(str(ws / args[0])))
            if not files:
                state["log"].append({"junit": args[0], "error": "No test report files were found"})
                mark("FAILURE")
                return "FAILURE"
            cases = read_junit(files)
            state["tests"] = cases
            if any(not c["ok"] for c in cases):
                mark("UNSTABLE")
                return "UNSTABLE"
            return "SUCCESS"
        if name == "archiveArtifacts":
            files = sorted(glob.glob(str(ws / args[0])))
            if not files:  # 문서: "Normally, a build fails if archiving returns zero artifacts."
                state["log"].append({"archiveArtifacts": args[0], "error": "No artifacts found"})
                mark("FAILURE")
                return "FAILURE"
            if archive_dir:
                Path(archive_dir).mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copy(f, Path(archive_dir) / Path(f).name)
            state["archived"] = [str(Path(f).relative_to(ws)) for f in files]
            return "SUCCESS"
        raise NotImplementedError(f"지원하지 않는 단계: {name}")

    stages = []
    for st in p["stages"]:
        row = {"name": st["name"], "status": None, "seconds": 0.0}
        if state["result"] == "FAILURE":
            row["status"] = "SKIPPED(earlier failure)"
        elif state["result"] == "UNSTABLE" and "skipStagesAfterUnstable" in p["options"]:
            row["status"] = "SKIPPED(unstable)"
        elif st["when"] is not None and not eval_when(st["when"][0][2], state["result"]):
            row["status"] = "SKIPPED(when)"
        else:
            t0, status = time.perf_counter(), "SUCCESS"
            for name, args, _ in st["steps"]:
                s = do_step(name, args)
                if WORSE[s] > WORSE[status]:
                    status = s
                if s == "FAILURE":
                    break
            row["status"], row["seconds"] = status, time.perf_counter() - t0
        stages.append(row)

    final = state["result"] or "SUCCESS"
    post_ran = []
    for cond in ("always", "success", "unstable", "failure"):
        if cond in p["post"] and (cond == "always" or cond.upper() == final):
            for name, args, _ in p["post"][cond]:
                do_step(name, args)
            post_ran.append(cond)
    final = state["result"] or "SUCCESS"
    return {"result": final, "stages": stages, "tests": state["tests"], "archived": state["archived"],
            "post_ran": post_ran, "log": state["log"]}


def lint(text):
    """이 장에서 정한 규칙을 검사한다. 문제 목록(빈 목록이면 통과)을 돌려준다.
    1) post { always { junit ... } }가 있어 실패해도 테스트 보고서가 남는다.
    2) 테스트 실패를 `|| true`로 삼키는 sh가 있으면, 같은 단계에서 바로 junit으로 결과를 읽어야 한다.
    3) push 단계는 테스트 단계들보다 뒤에 있고, 실패·UNSTABLE에서 돌지 않게 막혀 있어야 한다
       (when의 currentBuild.result 검사 또는 options { skipStagesAfterUnstable() }).
    """
    p = parse(text)
    problems = []
    always = [n for n, a, b in p["post"].get("always", [])]
    if "junit" not in always:
        problems.append("post { always { junit ... } }가 없다: 실패한 빌드의 테스트 보고서가 남지 않는다")
    for st in p["stages"]:
        names = [(n, a) for n, a, b in st["steps"]]
        for i, (n, a) in enumerate(names):
            if n == "sh" and "|| true" in a[0] and "pytest" in a[0]:
                if not any(m == "junit" for m, _ in names[i + 1:]):
                    problems.append(f"'{st['name']}': 테스트 실패를 || true로 삼키고 같은 단계에서 junit으로 읽지 않는다")
    names = [st["name"] for st in p["stages"]]
    push = [i for i, st in enumerate(p["stages"]) if "push" in st["name"].lower() or "deploy" in st["name"].lower()]
    tests = [i for i, st in enumerate(p["stages"]) if "test" in st["name"].lower()]
    for i in push:
        if tests and i < max(tests):
            problems.append(f"'{names[i]}'가 테스트 단계보다 앞에 있다")
        st = p["stages"][i]
        guarded = st["when"] is not None and any(v == "currentBuild.result" for _, v in st["when"][0][2])
        if not guarded and "skipStagesAfterUnstable" not in p["options"]:
            problems.append(f"'{names[i]}'가 UNSTABLE 빌드에서도 돈다(when 검사도 skipStagesAfterUnstable도 없음)")
    return problems
