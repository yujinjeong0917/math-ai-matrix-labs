"""모든 장의 테스트를 장 폴더마다 따로 돌린다(폴더마다 같은 이름의 모듈이 있어 한꺼번에 모으면 충돌한다).
- numpy·torch가 필요한 장: 설치돼 있으면 pytest, 없으면 '건너뜀'으로 센다.
- 표준 라이브러리만 쓰는 장: python3 -m unittest.
사용: python3 scripts/run_all_tests.py [트랙 ...]   (예: python3 scripts/run_all_tests.py aws or)
설치된 환경에서 전부 돌리려면: uv sync && uv run python scripts/run_all_tests.py"""
import importlib.util, subprocess, sys
from pathlib import Path
import sync_chapter_readmes as S

ROOT = S.ROOT
have_heavy = all(importlib.util.find_spec(m) for m in ("numpy", "torch"))
have_pytest = importlib.util.find_spec("pytest") is not None
tracks = sys.argv[1:] or list(S.NAMES)
ok = fail = skipped = 0
for t in tracks:
    for d in sorted((ROOT / t).glob("ch*")):
        tests = list(d.glob("test_*.py"))
        if not tests:
            continue
        runner = S.test_runner(d)
        heavy = S.needs_heavy_deps(d)
        if (heavy and not (have_heavy and have_pytest)) or (runner == "pytest" and not have_pytest):
            skipped += 1
            print(f"건너뜀  {d.relative_to(ROOT)} (uv 환경(numpy·torch·pytest) 필요)")
            continue
        cmd = [sys.executable, "-m", "pytest", "-q", "-x"] if runner == "pytest" else [sys.executable, "-m", "unittest", "-q"]
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        if r.returncode == 0:
            ok += 1
        else:
            fail += 1
            print(f"실패    {d.relative_to(ROOT)}\n{(r.stdout + r.stderr)[-800:]}")
print(f"통과 {ok} · 실패 {fail} · 건너뜀 {skipped}")
sys.exit(1 if fail else 0)
