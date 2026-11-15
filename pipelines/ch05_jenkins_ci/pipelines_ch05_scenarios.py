"""5장 실험과 테스트가 함께 쓰는 도우미: 브랜치를 새 작업 공간에 받아 파이프라인 돌리기, Jenkinsfile 변형 만들기.

experiments.py라는 이름은 장마다 있어서, 테스트가 그 이름을 import하지 않도록 따로 뺐다.
"""

import subprocess
from pathlib import Path

import pipelines_ch05_jenkins as jk
import pipelines_ch05_project as pr


def clone(repo, branch, dest):
    subprocess.run(["git", "clone", "-q", "-b", branch, str(repo), str(dest)], check=True, env=pr.git_env())
    return pr.git(dest, "rev-parse", "HEAD").stdout.strip()


def run_at(repo, branch, root, name, jenkinsfile=None):
    ws = Path(root) / name
    sha = clone(repo, branch, ws)
    reg = Path(root) / f"registry-{name}"
    # GIT_COMMIT과 REGISTRY_DIR를 환경변수로 넘겨 ci/push.py가 작업 공간 밖 폴더에 기록하게 한다
    r = jk.run_pipeline(jenkinsfile or (ws / "Jenkinsfile").read_text(), ws,
                        {"GIT_COMMIT": sha, "REGISTRY_DIR": str(reg)}, archive_dir=Path(root) / f"archive-{name}")
    r["sha"] = sha
    r["pushed"] = sorted(p.name for p in reg.glob("*.json")) if reg.exists() else []
    # 결과 비교에 필요 없는 단계별 명령 로그는 뺀다
    r.pop("log")
    return r


def short(r):
    return {"sha": r["sha"][:12], "result": r["result"], "stages": [(s["name"], s["status"]) for s in r["stages"]],
            "n_tests": len(r["tests"]), "failed": [t for t in r["tests"] if not t["ok"]], "pushed": r["pushed"],
            "post_ran": r["post_ran"], "archived": r["archived"]}


JF = pr.jenkinsfile_text()
# Jenkinsfile의 해당 sh 줄과 글자 하나까지 같아야 variants()의 replace가 동작한다. 다르면 아래 assert가 실패해 알려 준다
UNIT = "sh 'python -m pytest -q tests --ignore=tests/test_model.py --junitxml=reports/unit.xml'"
PERF = "sh 'python -m pytest -q tests/test_model.py --junitxml=reports/model.xml'"
# push 단계의 when 가드와 skipStagesAfterUnstable 옵션 원문. 변형 Jenkinsfile에서 이 둘을 빼는 데 쓴다
GUARD = """      when {
        expression { currentBuild.result == null || currentBuild.result == 'SUCCESS' }
      }
"""
OPTS = """  options {
    skipStagesAfterUnstable()
  }
"""


def variants():
    # 변형 B~E: 테스트 명령 뒤에 `|| true`를 붙여 실패 종료 코드를 삼킨다
    swallow = (JF.replace(UNIT, UNIT[:-1] + " || true'").replace(PERF, PERF[:-1] + " || true'")
               .replace("sh 'python -m churn.train'", "sh 'python -m churn.train || true'"))
    junit_in_stage = (swallow.replace(UNIT[:-1] + " || true'", UNIT[:-1] + " || true'\n        junit 'reports/unit.xml'")
                      .replace(PERF[:-1] + " || true'", PERF[:-1] + " || true'\n        junit 'reports/model.xml'"))
    v = {
        # A는 이 장의 Jenkinsfile 그대로. B: junit을 post에서만 읽음, C: 같은 단계에서 junit, D: C에서 가드까지 제거, E: junit 없음
        "A_ours": JF,
        "B_true_junit_post_only_guard": swallow.replace(OPTS, ""),
        "C_true_junit_in_stage_guard": junit_in_stage.replace(OPTS, ""),
        "D_true_junit_in_stage_no_guard": junit_in_stage.replace(OPTS, "").replace(GUARD, ""),
        "E_true_no_junit": swallow.replace(OPTS, "").replace("      junit 'reports/*.xml'\n", ""),
    }
    for k, t in v.items():
        assert k == "A_ours" or t != JF
    return v
