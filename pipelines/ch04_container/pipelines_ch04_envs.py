"""4장 환경 도구: 환경을 '설치 방법'이 아니라 '설치된 결과의 목록'으로 다루고, 그 목록의 지문을 만든다.

- parse_pins: `uv pip compile` 출력(또는 requirements 파일)에서 이름==버전만 뽑는다.
- lock_digest: 정렬한 고정 목록의 SHA-256. 같은 목록이면 순서·주석·해시 줄과 무관하게 같다.
- env_snapshot / diff_env: 지금 파이썬 환경을 기록하고 두 기록을 비교한다(3장 env_snapshot의 확장).
- compile_at / run_isolated: uv로 날짜를 잘라 의존성을 풀어 보거나, 새 격리 환경에서 명령을 실행한다.
  네트워크와 uv가 필요하므로 experiments.py에서만 쓰고, 테스트는 이것 없이 돈다.
"""

import hashlib
import json
import os
import platform
import re
import subprocess
import time
from importlib import metadata

# `이름==버전` 한 줄만 인식하는 정규식. 범위 지정(>=)이나 URL 의존성은 인식하지 않는다
PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([A-Za-z0-9_.+!\-]+)")


# 패키지 이름 정규화: 구분자(-, _, .)를 -로 통일하고 소문자로(NumPy == numpy)
def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(text):
    pins = {}
    for line in text.splitlines():
        # 줄 끝 주석 제거
        line = line.split("#", 1)[0]
        m = PIN.match(line)
        if m:
            pins[normalize(m.group(1))] = m.group(2)
    return pins


def lock_text(pins):
    # 이름 순으로 정렬해 같은 고정 목록이 항상 같은 텍스트가 되게 한다(지문의 입력)
    return "".join(f"{k}=={pins[k]}\n" for k in sorted(pins))


def lock_digest(pins):
    return "sha256:" + hashlib.sha256(lock_text(pins).encode()).hexdigest()


def env_snapshot(packages=("numpy", "torch", "pytest")):
    out = {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system()}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def diff_env(a, b):
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def _child_env():
    env = dict(os.environ)
    # 스레드 수를 2로 고정해 실행 시간·수치 비교를 안정화
    env.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "UV_NO_PROGRESS": "1"})
    # 현재 가상환경을 자식 프로세스가 이어받으면 격리 실행이 섞이므로 제거
    env.pop("VIRTUAL_ENV", None)
    return env


def compile_at(requirements, cutoff, python="3.12"):
    """requirements(문자열)를 cutoff(UTC 날짜) 이전에 올라온 배포판만으로 푼다."""
    r = subprocess.run(
        # --exclude-newer: 이 시각 이후에 올라온 배포판은 후보에서 뺀다. 과거 날짜의 환경을 다시 풀어 보는 용도
        ["uv", "pip", "compile", "-", "--python-version", python, "--exclude-newer", f"{cutoff}T00:00:00Z", "-q", "--no-header"],
        input=requirements, capture_output=True, text=True, env=_child_env(), timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return parse_pins(r.stdout)


def run_isolated(pins, args, python="3.12", cwd=None):
    """pins만 깔린 새 격리 환경에서 python args를 실행한다. (반환코드, stdout, stderr, 걸린 초)"""
    # --isolated: 현재 환경과 무관한 새 격리 환경. pins에 적힌 버전만 --with로 설치
    cmd = ["uv", "run", "-q", "--isolated", "--no-project", "--python", python]
    for k in sorted(pins):
        cmd += ["--with", f"{k}=={pins[k]}"]
    cmd += ["python", *args]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=_child_env(), cwd=cwd, timeout=600)
    return r.returncode, r.stdout, r.stderr, time.perf_counter() - t0


def last_json_line(stdout):
    # 경고 등 다른 출력이 섞여도 마지막 JSON 줄만 뽑는다
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError("JSON 줄이 없다")
