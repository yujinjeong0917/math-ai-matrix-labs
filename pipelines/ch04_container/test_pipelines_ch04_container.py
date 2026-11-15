"""4장 검증. `uv run pytest -q pipelines/ch04_container` 로 실행한다. 네트워크와 Docker 없이 돈다."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pipelines_ch04_data as dat
import pipelines_ch04_envs as envs
import pipelines_ch04_image as img
import pipelines_ch04_job as job
import pipelines_ch04_model_np as mnp

HERE = Path(__file__).parent
DOCKERFILE = (HERE / "Dockerfile").read_text()
# Dockerfile의 설치 RUN 줄과 정확히 같아야 한다(runners 딕셔너리의 키로 쓴다)
RUN_LOCK = "pip install --no-cache-dir --require-hashes -r requirements.lock"


# ---------- Dockerfile 규칙 ----------

def test_dockerfile_passes_lint():
    assert img.lint_dockerfile(DOCKERFILE) == []


@pytest.mark.parametrize("bad, expect", [
    # 각 줄: (규칙을 일부러 어긴 Dockerfile, 오류 메시지에 들어 있어야 할 단어)
    (DOCKERFILE.replace("python:3.12-slim@sha256:", "python:3.12-slim#"), "다이제스트"),
    ("FROM python:3.12-slim@sha256:aa\nCOPY . /app/\nRUN " + RUN_LOCK + "\nUSER 10001\nENTRYPOINT [\"python\"]\n", "뒤에"),
    (DOCKERFILE.replace("USER 10001", "USER root"), "root"),
    (DOCKERFILE.replace('ENTRYPOINT ["python", "pipelines_ch04_job.py"]', "ENTRYPOINT python pipelines_ch04_job.py"), "exec"),
    (DOCKERFILE.replace(" --require-hashes", ""), "해시"),
])
def test_lint_catches(bad, expect):
    assert any(expect in p for p in img.lint_dockerfile(bad))


def test_lock_has_hashes_and_exact_pins():
    text = (HERE / "requirements.lock").read_text()
    pins = envs.parse_pins(text)
    assert pins["numpy"] == "2.3.3" and pins["pytest"] == "8.4.2"
    # 패키지마다 최소 한 개의 파일 해시가 있어야 한다(--require-hashes 설치의 전제)
    assert text.count("--hash=sha256:") >= len(pins)


# ---------- 환경 지문 ----------

def test_lock_digest_ignores_order_comments_hashes():
    a = "numpy==2.3.3\npytest==8.4.2\n"
    # 순서, 주석, 해시 줄, 대소문자가 달라도 같은 고정 목록이면 같은 지문이어야 한다
    b = "# via x\npytest==8.4.2 \\\n    --hash=sha256:abc\nNumPy==2.3.3  # 주석\n"
    assert envs.lock_digest(envs.parse_pins(a)) == envs.lock_digest(envs.parse_pins(b))
    assert envs.lock_digest(envs.parse_pins(a)) != envs.lock_digest(envs.parse_pins(a.replace("2.3.3", "2.3.2")))


def test_env_snapshot_and_diff():
    s = envs.env_snapshot()
    assert s["numpy"] == np.__version__
    assert envs.diff_env(s, dict(s)) == {}
    assert envs.diff_env(s, {**s, "numpy": "1.26.0"}) == {"numpy": (np.__version__, "1.26.0")}


# ---------- 내용 주소 빌더 ----------

# 가짜 베이스 이미지와 레지스트리, 설치 횟수를 세는 가짜 RUN 실행기를 만든다. 반환: (레지스트리, Dockerfile, 빌드 컨텍스트, runners, 횟수, 베이스)
def _setup(base_bytes=b"py day1"):
    base = img.make_base("python:3.12-slim", {"usr/bin/python": (base_bytes, 1)}, created=1)
    reg = img.Registry()
    reg.push("python:3.12-slim", base)
    count = {"install": 0}

    def install(fs, ctx):
        count["install"] += 1
        return {f"site-packages/{k}-{v}.dist-info/METADATA": b"x" for k, v in envs.parse_pins(ctx["requirements.lock"][0].decode()).items()}

    df = DOCKERFILE.replace("python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f", "python:3.12-slim@" + base["digest"])
    ctx = {p: ((HERE / p).read_bytes(), 100) for p in ["requirements.lock", "pipelines_ch04_data.py", "pipelines_ch04_model_np.py", "pipelines_ch04_job.py"]}
    return reg, df, ctx, {RUN_LOCK: install}, count, base


def test_same_inputs_same_digest():
    reg, df, ctx, runners, _, _ = _setup()
    a = img.build(df, ctx, reg, runners, now=5, epoch=0)
    b = img.build(df, ctx, reg, runners, now=99, epoch=0)
    assert a["digest"] == b["digest"]
    # epoch를 안 주면 빌드 시각이 설정에 들어가 다이제스트가 달라진다
    c = img.build(df, ctx, reg, runners, now=99)  # 시각을 고정하지 않으면 달라진다
    assert c["digest"] != a["digest"]


def test_source_edit_does_not_reinstall():
    reg, df, ctx, runners, count, _ = _setup()
    cache = {}
    img.build(df, ctx, reg, runners, cache=cache)
    ctx2 = {**ctx, "pipelines_ch04_job.py": (ctx["pipelines_ch04_job.py"][0] + b"\n# edit\n", 200)}
    b = img.build(df, ctx2, reg, runners, cache=cache)
    assert count["install"] == 1
    # 캐시에서 나온 단계는 lock 복사와 설치 두 개(소스 복사만 다시 실행)
    assert [how for _, how, _ in b["log"]].count("cache") == 2


def test_copy_cache_ignores_mtime_but_layer_bytes_do_not():
    reg, df, ctx, runners, count, _ = _setup()
    cache = {}
    a = img.build(df, ctx, reg, runners, cache=cache, now=1, epoch=0)
    # 내용은 그대로 두고 수정 시각만 1시간 뒤로 바꾼 컨텍스트
    touched = {k: (v[0], v[1] + 3600) for k, v in ctx.items()}
    b = img.build(df, touched, reg, runners, cache=cache, now=1, epoch=0)
    assert all(how != "copy" for _, how, _ in b["log"]) and a["digest"] == b["digest"]
    c = img.build(df, touched, reg, runners, cache=None, now=1)  # 캐시 없이, 시각 고정 없이
    d = img.build(df, ctx, reg, runners, cache=None, now=1)
    # 캐시도 epoch도 없으면 수정 시각이 tar 바이트에 들어가 다이제스트가 달라진다
    assert c["digest"] != d["digest"]


def test_run_cache_is_stale_by_design():
    """RUN은 명령 문자열만 보므로, 바깥 세상이 바뀌어도 캐시가 옛 결과를 준다(Docker 문서와 같은 동작)."""
    reg, df, ctx, _, _, _ = _setup()
    world = {"numpy": "1.26.4"}
    loose = df.replace("COPY requirements.lock /app/\n", "").replace(RUN_LOCK, "pip install numpy")
    runners = {"pip install numpy": lambda fs, c: {f"usr/lib/python3.12/site-packages/numpy-{world['numpy']}.dist-info/M": b"x"}}
    cache = {}
    img.build(loose, ctx, reg, runners, cache=cache)
    world["numpy"] = "2.0.0"
    # 캐시를 쓰면 옛 numpy(1.26.4), 캐시 없이 다시 지으면 새 값(2.0.0)이 나온다
    assert img.installed_versions(img.build(loose, ctx, reg, runners, cache=cache)["fs"])["numpy"] == "1.26.4"
    assert img.installed_versions(img.build(loose, ctx, reg, runners, cache=None)["fs"])["numpy"] == "2.0.0"


def test_tag_moves_digest_does_not():
    reg, df, ctx, runners, _, base1 = _setup()
    base2 = img.make_base("python:3.12-slim", {"usr/bin/python": (b"py day2", 2)}, created=2)
    reg.push("python:3.12-slim", base2)
    assert reg.resolve("python:3.12-slim") == base2["digest"]
    assert reg.resolve("python:3.12-slim@" + base1["digest"]) == base1["digest"]
    by_tag = df.replace("python:3.12-slim@" + base1["digest"], "python:3.12-slim")
    assert img.build(by_tag, ctx, reg, runners, epoch=0)["config"]["base"] == base2["digest"]
    assert img.build(df, ctx, reg, runners, epoch=0)["config"]["base"] == base1["digest"]
    with pytest.raises(KeyError):
        reg.resolve("python:3.12-slim@sha256:" + "0" * 64)


def test_tar_layer_deterministic():
    f = {"b.txt": (b"2", 5), "a.txt": (b"1", 7)}
    assert img.tar_layer(f) == img.tar_layer(dict(reversed(list(f.items()))))
    assert img.tar_layer(f) != img.tar_layer({k: (v[0], v[1] + 1) for k, v in f.items()})
    assert img.tar_layer(f, epoch=0) == img.tar_layer({k: (v[0], v[1] + 1) for k, v in f.items()}, epoch=0)


# ---------- 작업 스크립트 ----------

def test_job_matches_chapter3():
    """3장 results의 데이터 지문 앞자리와 정확도(같은 시드·설정)와 같다."""
    table, tr, model, metrics = job.train()
    # 3장 결과(results/ch03.json)의 데이터 지문 앞 12자리와 정확도 0.811(검증 1000행 기준 0.001 단위). 아래 docstring이 이 대응을 말한다
    assert job.table_hash(table).startswith("a21adb9b4db6")
    assert metrics["val_acc"] == pytest.approx(0.811)


def test_nep50_in_this_env():
    major = int(np.__version__.split(".")[0])
    stats = mnp.serving_stats(dat.make_table(1, 200))
    t32 = {k: v.astype(np.float32) for k, v in dat.make_table(2, 50).items() if k in dat.NUMERIC}
    dtype, _ = mnp.transform_serving(t32, stats)
    # NumPy 2.x면 float64, 1.x면 float32가 나와야 한다(NEP 50)
    assert dtype == ("float64" if major >= 2 else "float32")


def test_job_save_load_roundtrip(tmp_path):
    out = subprocess.run([sys.executable, str(HERE / "pipelines_ch04_job.py"), "--save", str(tmp_path), "--load", str(tmp_path)],
                         capture_output=True, text=True, check=True).stdout
    r = envs.last_json_line(out)
    assert r["load"]["pkl"] == {"ok": True, "same_w": True}
    assert r["load"]["npz"] == {"ok": True, "same_w": True}


def test_numpy_and_torch_agree():
    # torch가 없는 환경에서는 이 테스트를 건너뛴다
    torch = pytest.importorskip("torch")
    import pipelines_ch04_model_torch as mtt
    torch.set_num_threads(2)
    table = dat.make_table(3, 1500)
    tr, va = dat.split(table, 1000, 7)
    m1, s1 = mnp.train_and_eval(tr, va, 0.5, 200, 0.01)
    m2, s2 = mnp.train_and_eval(tr, va, 0.5, 200, 0.01, fit=mtt.fit_logistic)
    assert np.max(np.abs(m1["w"] - m2["w"])) < 1e-12 and s1 == s2


def test_results_file_consistent():
    p = HERE / "results" / "ch04.json"
    if not p.exists():
        pytest.skip("experiments.py를 아직 돌리지 않았다")
    r = json.loads(p.read_text())
    # experiments.py를 이미 돌려 results/ch04.json이 있을 때만 결과 파일 내부의 일관성을 확인
    rows = r["E1"]["rows"]
    assert rows["top_only_day1"]["import_ok"] and not rows["top_only_day2"]["import_ok"] and rows["lock_day2"]["import_ok"]
    assert rows["top_only_day1"]["lock_digest"] == rows["lock_day2"]["lock_digest"]
    assert r["E5"]["deps_before_source"]["installs_during_edits"] == 0
    assert r["E6"]["digest_lock_epoch"]["same_digest_fresh"]
