"""4장 비교 실험. `uv run python pipelines/ch04_container/experiments.py` 로 실행한다(네트워크와 uv 필요).

E1 날짜만 다른 두 설치: 맨 위 패키지 하나만 고정한 목록(onnxruntime==1.18.0)과, 첫날 풀어 둔 전체 lock을
   2024-06-10, 2024-06-20 두 날짜 기준으로 각각 풀고(uv --exclude-newer), 그 환경에서 import해 본다.
E2 같은 작업, 다른 NumPy: 3장과 같은 데이터·설정의 학습을 지금 환경(2.3.3)과 격리 환경 4개
   (2.3.3, 2.0.0, 1.26.4, 1.26.0)에서 돌려 데이터 지문·가중치·정확도·서빙 확률을 비교한다.
E3 저장한 모델 파일: 2.3.3에서 pickle과 npz로 저장하고, 각 환경에서 불러 본다.
E4 설치 비용 실측: 새 가상환경에 lock을 설치하는 시간(uv 캐시 사용/미사용)과 설치된 크기.
E5 층 순서(시뮬레이션): 소스를 10번 고칠 때, 의존성 설치를 소스 복사 앞에 둔 Dockerfile과 뒤에 둔 Dockerfile의 재설치 횟수.
E6 이틀 간격 빌드(시뮬레이션): 태그 vs 다이제스트, 느슨한 목록 vs lock, 캐시 유무, 시각 고정 유무에 따라 이미지 다이제스트가 같은지.
E7 NumPy와 PyTorch 대조: 고정한 환경 안에서 두 구현의 가중치 차이.

결과는 results/ch04.json 에 저장하고, 웹 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pipelines_ch04_data as dat  # noqa: E402
import pipelines_ch04_envs as envs  # noqa: E402
import pipelines_ch04_image as img  # noqa: E402
import pipelines_ch04_job as job  # noqa: E402
import pipelines_ch04_model_np as mnp  # noqa: E402
import pipelines_ch04_model_torch as mtt  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch04.json"
JOB = str(HERE / "pipelines_ch04_job.py")
DAY1, DAY2 = "2024-06-10", "2024-06-20"
TOP_ONLY = "onnxruntime==1.18.0\n"
NUMPY_ENVS = ["2.3.3", "2.0.0", "1.26.4", "1.26.0"]
IMPORT_PROBE = (
    "import json, numpy\n"
    "try:\n"
    "    import onnxruntime\n"
    "    r = {'ok': True, 'onnxruntime': onnxruntime.__version__}\n"
    "except Exception as e:\n"
    "    r = {'ok': False, 'error': type(e).__name__ + ': ' + str(e)}\n"
    "r['numpy'] = numpy.__version__\n"
    "print(json.dumps(r))\n"
)


def uv_version():
    return subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip()


def e1_two_days():
    out = {"day1": DAY1, "day2": DAY2, "requirements_top_only": TOP_ONLY.strip()}
    top1, top2 = envs.compile_at(TOP_ONLY, DAY1), envs.compile_at(TOP_ONLY, DAY2)
    lock1 = envs.lock_text(top1)
    lock_on_day2 = envs.compile_at(lock1, DAY2)
    loose = {d: envs.compile_at("numpy>=1.24\n", d)["numpy"] for d in (DAY1, DAY2, "2026-10-08")}
    rows = {}
    for name, pins in (("top_only_day1", top1), ("top_only_day2", top2), ("lock_day2", lock_on_day2)):
        code, so, se, sec = envs.run_isolated(pins, ["-c", IMPORT_PROBE])
        probe = envs.last_json_line(so) if so.strip() else {"ok": False, "error": se.strip().splitlines()[-1]}
        warn = next((ln for ln in se.splitlines() if "compiled using NumPy 1.x" in ln), None)
        rows[name] = {"pins": pins, "n_packages": len(pins), "numpy": pins.get("numpy"), "lock_digest": envs.lock_digest(pins),
                      "import_ok": probe["ok"], "error": probe.get("error"), "stderr_first_line": warn, "seconds": sec}
    out["rows"] = rows
    out["changed_packages_top_only"] = {k: (top1.get(k), top2.get(k)) for k in sorted(set(top1) | set(top2)) if top1.get(k) != top2.get(k)}
    out["lock_same_on_both_days"] = lock_on_day2 == top1
    out["loose_numpy_ge_1_24"] = loose
    return out


def e2_e3_numpy_envs():
    res = {}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        r = subprocess.run([sys.executable, JOB, "--save", str(d / "model"), "--dump-p", str(d / "p_lab.npy"), "--dump-table", str(d / "t_lab.npz")],
                           capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "2"}, check=True)
        lab = envs.last_json_line(r.stdout)
        p_lab = np.load(d / "p_lab.npy")
        res["lab"] = {k: v for k, v in lab.items() if k != "w"} | {"env": envs.env_snapshot()}
        for v in NUMPY_ENVS:
            envs.run_isolated({"numpy": v}, ["-c", "import numpy"])  # 첫 실행: 내려받기·환경 준비
            code, so, se, sec = envs.run_isolated({"numpy": v}, [JOB, "--load", str(d / "model"), "--dump-p", str(d / f"p_{v}.npy"), "--dump-table", str(d / f"t_{v}.npz"), "--train-on", str(d / "t_lab.npz")])
            if code != 0:
                res[v] = {"returncode": code, "stderr_tail": se.strip().splitlines()[-3:]}
                continue
            o = envs.last_json_line(so)
            p = np.load(d / f"p_{v}.npy")
            ta, tb = np.load(d / "t_lab.npz"), np.load(d / f"t_{v}.npz")
            col_diff = {k: {"n_cells": int(np.sum(ta[k] != tb[k])), "max_abs": float(np.max(np.abs(ta[k] - tb[k])))}
                        for k in ta.files if not np.array_equal(ta[k], tb[k])}
            w_diff = float(np.max(np.abs(np.array(o["w"]) - np.array(lab["w"]))))
            res[v] = {k: val for k, val in o.items() if k != "w"} | {
                "seconds_second_run": sec,
                "same_data_hash": o["data_hash"] == lab["data_hash"], "data_col_diff": col_diff,
                "w_bitwise_same_on_lab_table": o["w_sha256_on_given"] == lab["w_sha256"],
                "w_max_abs_diff": w_diff, "w_bitwise_same": o["w_sha256"] == lab["w_sha256"],
                "serve_p_max_abs_diff": float(np.max(np.abs(p - p_lab))),
                "serve_p_n_diff": int(np.sum(p != p_lab)), "serve_n": int(p.size),
                "serve_label_flips": int(np.sum((p >= 0.5) != (p_lab >= 0.5))),
                "serve_min_dist_to_half": float(np.min(np.abs(p_lab - 0.5))),
                "env_diff_vs_lab": {"python": (lab["python"], o["python"]), "numpy": (lab["numpy"], o["numpy"])},
            }
    return res


def e4_install_cost(reps=3):
    lock = (HERE / "requirements.lock").read_text()
    pins = envs.parse_pins(lock)
    plain = envs.lock_text(pins)  # 해시 줄은 플랫폼(리눅스)용이라, 이 컴퓨터에서는 같은 버전 목록만 쓴다
    rows = {}
    for mode in ("warm_cache", "no_cache"):
        ts, size = [], None
        for _ in range(reps):
            with tempfile.TemporaryDirectory() as d:
                venv = Path(d) / "venv"
                subprocess.run(["uv", "venv", "-q", "--python", "3.12", str(venv)], check=True, env=envs._child_env())
                req = Path(d) / "req.txt"
                req.write_text(plain)
                cmd = ["uv", "pip", "install", "-q", "--python", str(venv / "bin" / "python"), "-r", str(req)]
                if mode == "no_cache":
                    cmd.append("--no-cache")
                t0 = time.perf_counter()
                subprocess.run(cmd, check=True, env=envs._child_env())
                ts.append(time.perf_counter() - t0)
                sp = next(venv.glob("lib/python3.12/site-packages"))
                size = sum(f.stat().st_size for f in sp.rglob("*") if f.is_file())
        rows[mode] = {"seconds_each": ts, "seconds_median": statistics.median(ts), "site_packages_bytes": size}
    return {"pins": pins, "rows": rows}


# ---------- 시뮬레이션 공용 ----------

GOOD = (HERE / "Dockerfile").read_text()
BAD = """FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
WORKDIR /app
COPY . /app/
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
USER 10001
ENTRYPOINT ["python", "pipelines_ch04_job.py"]
"""
CONTEXT_FILES = ["Dockerfile", "requirements.lock", "pipelines_ch04_data.py", "pipelines_ch04_model_np.py", "pipelines_ch04_job.py"]


def context_now(mtime=1_728_000_000):
    return {p: ((HERE / p).read_bytes(), mtime) for p in CONTEXT_FILES}


def install_from_lock_runner(counter):
    def run(fs, ctx):
        counter["install"] += 1
        pins = envs.parse_pins(ctx["requirements.lock"][0].decode())
        return {f"usr/lib/python3.12/site-packages/{k}-{v}.dist-info/METADATA": f"{k}=={v}\n".encode() for k, v in pins.items()}
    return run


def base_images():
    day1 = img.make_base("python:3.12-slim", {"usr/bin/python3.12": (b"cpython 3.12 (day1 patch)", 1)}, created=1)
    day2 = img.make_base("python:3.12-slim", {"usr/bin/python3.12": (b"cpython 3.12 (day2 patch + security fix)", 2)}, created=2)
    return day1, day2


def e5_layer_order(n_edits=10):
    day1, _ = base_images()
    out = {}
    for name, df in (("deps_before_source", GOOD), ("deps_after_source", BAD)):
        reg = img.Registry()
        reg.push("python:3.12-slim", day1)
        df = df.replace("python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f", "python:3.12-slim@" + day1["digest"])
        cnt = {"install": 0}
        runners = {"pip install --no-cache-dir --require-hashes -r requirements.lock": install_from_lock_runner(cnt)}
        cache = {}
        ctx = context_now()
        img.build(df, ctx, reg, runners, cache=cache, now=100)
        first = cnt["install"]
        rebuilt_steps, new_bytes = [], 0
        for i in range(n_edits):
            src, m = ctx["pipelines_ch04_job.py"]
            ctx = {**ctx, "pipelines_ch04_job.py": (src + f"\n# edit {i}\n".encode(), m + i + 1)}
            b = img.build(df, ctx, reg, runners, cache=cache, now=200 + i)
            rebuilt_steps.append(sum(1 for _, how, _ in b["log"] if how in ("run", "copy")))
            new_bytes += sum(sz for _, how, sz in b["log"] if how in ("run", "copy"))
        out[name] = {"installs_first_build": first, "installs_during_edits": cnt["install"] - first,
                     "rebuilt_steps_per_edit": rebuilt_steps, "new_layer_bytes_sim": new_bytes,
                     "lint_problems": img.lint_dockerfile(df.replace("python:3.12-slim@" + day1["digest"], "python:3.12-slim@sha256:x"))}
    out["n_edits"] = n_edits
    return out


def e6_two_days(resolved_day1, resolved_day2):
    """resolved_dayN: E1에서 실제로 푼 결과(맨 위 패키지만 고정한 목록이 그날 무엇으로 풀렸나)."""
    b1, b2 = base_images()
    loose_req = TOP_ONLY.encode()
    lock_bytes = envs.lock_text(resolved_day1).encode()  # 첫날 풀어 둔 전체 목록
    variants = {
        "tag_loose": ("python:3.12-slim", "loose", None),
        "digest_loose": ("python:3.12-slim@" + b1["digest"], "loose", None),
        "tag_lock": ("python:3.12-slim", "lock", None),
        "digest_lock": ("python:3.12-slim@" + b1["digest"], "lock", None),
        "digest_lock_epoch": ("python:3.12-slim@" + b1["digest"], "lock", 0),
    }
    out = {}
    for name, (ref, kind, epoch) in variants.items():
        req_name = "requirements.txt" if kind == "loose" else "requirements.lock"
        df = (f"FROM {ref}\nWORKDIR /app\nCOPY {req_name} /app/\nRUN pip install -r {req_name}\n"
              f"COPY pipelines_ch04_job.py /app/\nUSER 10001\nENTRYPOINT [\"python\", \"pipelines_ch04_job.py\"]\n")

        def runner_for(day_pins):
            def run(fs, ctx):
                pins = envs.parse_pins(ctx[req_name][0].decode()) if kind == "lock" else day_pins
                return {f"usr/lib/python3.12/site-packages/{k}-{v}.dist-info/METADATA": f"{k}=={v}\n".encode() for k, v in pins.items()}
            return run

        builds, cache = {}, {}
        for day, base_now, pins, now in (("day1", b1, resolved_day1, 1_718_000_000), ("day2", b2, resolved_day2, 1_718_864_000)):
            reg = img.Registry()
            reg.push("python:3.12-slim", b1)  # 첫날 이미지는 다이제스트로 계속 받을 수 있다
            reg.push("python:3.12-slim", base_now)  # 태그는 그날의 이미지를 가리킨다
            ctx = {req_name: (loose_req if kind == "loose" else lock_bytes, now - 3600),
                   "pipelines_ch04_job.py": ((HERE / "pipelines_ch04_job.py").read_bytes(), now - 3600)}  # 그날 새로 체크아웃
            runners = {f"-r {req_name}".join(["pip install ", ""]): runner_for(pins)}
            fresh = img.build(df, ctx, reg, runners, cache=None, now=now, epoch=epoch)
            builds[day] = fresh
            if day == "day2":
                cached = img.build(df, ctx, reg, runners, cache=cache, now=now, epoch=epoch)
                builds["day2_cached"] = cached
            else:
                img.build(df, ctx, reg, runners, cache=cache, now=now, epoch=epoch)  # 첫날 캐시를 채운다
        out[name] = {
            "digest_day1": builds["day1"]["digest"], "digest_day2": builds["day2"]["digest"], "digest_day2_cached": builds["day2_cached"]["digest"],
            "same_digest_fresh": builds["day1"]["digest"] == builds["day2"]["digest"],
            "same_digest_cached": builds["day1"]["digest"] == builds["day2_cached"]["digest"],
            "numpy_day1": img.installed_versions(builds["day1"]["fs"]).get("numpy"),
            "numpy_day2": img.installed_versions(builds["day2"]["fs"]).get("numpy"),
            "numpy_day2_cached": img.installed_versions(builds["day2_cached"]["fs"]).get("numpy"),
            "same_files_fresh": {k: v[0] for k, v in builds["day1"]["fs"].items()} == {k: v[0] for k, v in builds["day2"]["fs"].items()},
            "day2_cached_log": [how for _, how, _ in builds["day2_cached"]["log"]],
        }
    return out


def e7_np_vs_torch():
    table = dat.make_table(job.DATA_SEED, job.N_ROWS)
    tr, va = dat.split(table, job.N_TRAIN, job.SPLIT_SEED)
    m_np, s_np = mnp.train_and_eval(tr, va, job.LR, job.STEPS, job.L2)
    m_t, s_t = mnp.train_and_eval(tr, va, job.LR, job.STEPS, job.L2, fit=mtt.fit_logistic)
    return {"w_max_abs_diff": float(np.max(np.abs(m_np["w"] - m_t["w"]))), "b_abs_diff": abs(m_np["b"] - m_t["b"]),
            "val_acc_np": s_np["val_acc"], "val_acc_torch": s_t["val_acc"]}


def main():
    t0 = time.perf_counter()
    res = {"env": envs.env_snapshot(), "uv": uv_version(), "checked": "2026-10-08"}
    res["E1"] = e1_two_days()
    print("E1", {k: (v["numpy"], v["import_ok"]) for k, v in res["E1"]["rows"].items()}, flush=True)
    res["E2_E3"] = e2_e3_numpy_envs()
    print("E2", {k: (v.get("val_acc"), v.get("w_max_abs_diff"), v.get("serve_dtype"), v.get("load")) for k, v in res["E2_E3"].items()}, flush=True)
    res["E4"] = e4_install_cost()
    print("E4", {k: (v["seconds_median"], v["site_packages_bytes"]) for k, v in res["E4"]["rows"].items()}, flush=True)
    res["E5"] = e5_layer_order()
    print("E5", {k: v if not isinstance(v, dict) else (v["installs_during_edits"], v["rebuilt_steps_per_edit"][:2]) for k, v in res["E5"].items()}, flush=True)
    rows = res["E1"]["rows"]
    res["E6"] = e6_two_days(rows["top_only_day1"]["pins"], rows["top_only_day2"]["pins"])
    print("E6", {k: (v["same_digest_fresh"], v["same_digest_cached"], v["numpy_day1"], v["numpy_day2"], v["numpy_day2_cached"]) for k, v in res["E6"].items()}, flush=True)
    res["E7"] = e7_np_vs_torch()
    print("E7", res["E7"], flush=True)
    res["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("saved", OUT, f"{res['total_seconds']:.1f}s")


if __name__ == "__main__":
    main()
