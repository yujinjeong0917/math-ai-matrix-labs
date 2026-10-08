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

PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([A-Za-z0-9_.+!\-]+)")


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(text):
    pins = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        m = PIN.match(line)
        if m:
            pins[normalize(m.group(1))] = m.group(2)
    return pins


def lock_text(pins):
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
    env.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "UV_NO_PROGRESS": "1"})
    env.pop("VIRTUAL_ENV", None)
    return env


def compile_at(requirements, cutoff, python="3.12"):
    """requirements(문자열)를 cutoff(UTC 날짜) 이전에 올라온 배포판만으로 푼다."""
    r = subprocess.run(
        ["uv", "pip", "compile", "-", "--python-version", python, "--exclude-newer", f"{cutoff}T00:00:00Z", "-q", "--no-header"],
        input=requirements, capture_output=True, text=True, env=_child_env(), timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return parse_pins(r.stdout)


def run_isolated(pins, args, python="3.12", cwd=None):
    """pins만 깔린 새 격리 환경에서 python args를 실행한다. (반환코드, stdout, stderr, 걸린 초)"""
    cmd = ["uv", "run", "-q", "--isolated", "--no-project", "--python", python]
    for k in sorted(pins):
        cmd += ["--with", f"{k}=={pins[k]}"]
    cmd += ["python", *args]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=_child_env(), cwd=cwd, timeout=600)
    return r.returncode, r.stdout, r.stderr, time.perf_counter() - t0


def last_json_line(stdout):
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError("JSON 줄이 없다")
