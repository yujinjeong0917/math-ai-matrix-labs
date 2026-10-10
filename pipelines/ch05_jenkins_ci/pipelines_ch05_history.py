"""2주 동안 4명이 같은 저장소를 고치는 상황을 진짜 git으로 만들고, 합치는 간격만 바꿔 비교한다.

저장소 모양(일부러 아주 작게 만들었다)
- schema.txt   : 데이터 열 이름, 한 줄에 하나(6개). 데이터 담당(dev0)만 고친다.
- registry.txt : 학습에 쓰는 특성 목록. 새 특성을 만들면 맨 끝에 한 줄 덧붙인다(모두가 같은 파일을 고친다).
- features/<id>.txt : "uses: <열 이름>" 한 줄. 그 특성이 읽는 열.
- config.txt   : 설정값 30줄. 아무나 한 줄씩 고친다.

하루에 사람마다 커밋 2개. dev0은 3일째와 7일째 첫 커밋에서 열 이름 하나를 바꾸고(<열>_v2),
자기 작업 사본에서 보이는 사용처를 모두 함께 고친다(브랜치 A와 같은 일).
나머지 커밋은 시드로 정한 동전 던지기로 "새 특성 추가" 또는 "설정 한 줄 수정"이다.

합치기(integration) 규칙: T일마다 하루가 끝날 때 dev0, dev1, dev2, dev3 순서로
  ① main을 자기 브랜치로 merge(텍스트 충돌이 나면 세고, registry는 양쪽 줄을 모두 남기고 config는 내 값을 남김)
  ② 검사(gate): registry의 모든 특성이 schema에 있는 열을 쓰는지 확인 → 깨진 특성 수를 세고, 고친 커밋을 하나 만든다
  ③ main을 자기 브랜치로 fast-forward
모두 합친 뒤에는 네 사람이 main을 다시 받는다.
branch_ci=True면 브랜치 커밋마다 같은 검사를 돌린다(브랜치 CI). 이때 검사가 몇 번 돌고 몇 번 실패하는지 센다.

검사 함수 gate()는 프로젝트 테스트가 하는 일("특성이 읽는 열이 데이터에 있나")을 파일만 보고 하는 축약판이다.
"""

import random
import re
from pathlib import Path

from pipelines_ch05_project import git

COLUMNS = ["tenure_months", "monthly_fee", "support_calls", "usage_hours", "age", "discount_rate"]
N_DEVS, DAYS, COMMITS_PER_DAY, N_CONFIG = 4, 10, 2, 30
RENAME_DAYS = (3, 7)


def _date(day, dev=0, k=0, minute=0):
    return f"2026-09-{day:02d}T{9 + dev:02d}:{10 * k + minute:02d}:00+09:00"


def _commit(wt, msg, date):
    git(wt, "add", "-A", date=date)
    git(wt, "commit", "-q", "-m", msg, date=date)


def read_state(wt):
    wt = Path(wt)
    schema = (wt / "schema.txt").read_text().split()
    registry = (wt / "registry.txt").read_text().split()
    uses = {}
    for fid in registry:
        p = wt / "features" / f"{fid}.txt"
        uses[fid] = p.read_text().split(":", 1)[1].strip() if p.exists() else None
    return schema, registry, uses


def gate(wt):
    """깨진 특성 목록: 읽는 열이 schema에 없거나 파일이 없는 특성."""
    schema, registry, uses = read_state(wt)
    return sorted(f for f in registry if uses[f] is None or uses[f] not in schema)


CONFLICT = re.compile(r"^<<<<<<< [^\n]*\n(.*?)^=======\n(.*?)^>>>>>>> [^\n]*\n", re.S | re.M)


def resolve(path):
    """충돌 표시를 풀고 (덩어리 수, 덩어리 안 줄 수: 양쪽 합)을 돌려준다. registry는 양쪽 줄을 모두, 그 밖은 내 쪽(ours)을 남긴다."""
    text = Path(path).read_text()
    blocks = CONFLICT.findall(text)
    n = (len(blocks), sum(len(a.splitlines()) + len(b.splitlines()) for a, b in blocks))
    if Path(path).name == "registry.txt":
        def keep(m):
            ours = m.group(1).splitlines(keepends=True)
            return "".join(ours + [ln for ln in m.group(2).splitlines(keepends=True) if ln not in ours])
    else:
        def keep(m):
            return m.group(1)
    Path(path).write_text(CONFLICT.sub(keep, text))
    return n


def setup(root):
    root = Path(root)
    main = root / "main"
    main.mkdir(parents=True)
    git(main, "init", "-q", "-b", "main")
    (main / "schema.txt").write_text("".join(c + "\n" for c in COLUMNS))
    (main / "features").mkdir()
    reg = []
    for i, c in enumerate(COLUMNS[:4]):
        fid = f"f_base_{i}"
        (main / "features" / f"{fid}.txt").write_text(f"uses: {c}\n")
        reg.append(fid)
    (main / "registry.txt").write_text("".join(f + "\n" for f in reg))
    (main / "config.txt").write_text("".join(f"p{i:02d} = 0\n" for i in range(N_CONFIG)))
    _commit(main, "base", _date(1, 0, 0))
    wts = []
    for d in range(N_DEVS):
        wt = root / f"dev{d}"
        git(main, "worktree", "add", "-q", "-b", f"dev{d}", str(wt), "main")
        wts.append(wt)
    return main, wts


def simulate(interval, seed=0, branch_ci=False, root=None):
    """interval: 며칠마다 합치나(1, 2, 5, 10). 돌려주는 값: 충돌·깨진 특성·발견까지 걸린 날 등."""
    rng = random.Random(seed)
    main, wts = setup(root)
    births = {}           # 특성 id -> 만든 날
    rename_day = {}       # 바뀐 옛 열 이름 -> 바뀐 날
    renamed_to = {}       # 옛 이름 -> 새 이름
    stats = {"interval": interval, "seed": seed, "commits": 0, "integrations": 0, "merges_with_conflict": 0,
             "conflict_hunks": 0, "conflict_lines": 0, "conflict_files": {}, "broken_found": 0, "fix_commits": 0, "delays": [],
             "branch_ci_runs": 0, "branch_ci_red": 0, "main_red_after_round": 0, "unsynced_changes_at_merge": []}
    since_sync = [0] * N_DEVS
    counter = 0
    for day in range(1, DAYS + 1):
        for d, wt in enumerate(wts):
            for k in range(COMMITS_PER_DAY):
                schema = (wt / "schema.txt").read_text().split()
                if d == 0 and day in RENAME_DAYS and k == 0:
                    old = rng.choice([c for c in schema if not c.endswith("_v2")])
                    new = old + "_v2"
                    (wt / "schema.txt").write_text("".join((new if c == old else c) + "\n" for c in schema))
                    for p in (wt / "features").glob("*.txt"):
                        if p.read_text().strip() == f"uses: {old}":
                            p.write_text(f"uses: {new}\n")
                    rename_day[old], renamed_to[old] = day, new
                    msg = f"rename {old} -> {new}"
                elif rng.random() < 0.5:
                    counter += 1
                    fid = f"f_{d}_{counter:03d}"
                    col = rng.choice(schema)
                    (wt / "features" / f"{fid}.txt").write_text(f"uses: {col}\n")
                    with open(wt / "registry.txt", "a") as fh:
                        fh.write(fid + "\n")
                    births[fid] = day
                    msg = f"add feature {fid} ({col})"
                else:
                    i = rng.randrange(N_CONFIG)
                    lines = (wt / "config.txt").read_text().splitlines(keepends=True)
                    lines[i] = f"p{i:02d} = {d}.{day}.{k}\n"
                    (wt / "config.txt").write_text("".join(lines))
                    msg = f"tune p{i:02d}"
                _commit(wt, msg, _date(day, d, k))
                stats["commits"] += 1
                since_sync[d] += 1
                if branch_ci:
                    stats["branch_ci_runs"] += 1
                    stats["branch_ci_red"] += bool(gate(wt))
        if day % interval == 0:
            stats["integrations"] += 1
            stats["unsynced_changes_at_merge"].append(sum(since_sync))
            for d, wt in enumerate(wts):
                r = git(wt, "merge", "-q", "--no-edit", "main", check=False, date=_date(day, d, 5, 1))
                conflicted = [f for f in git(wt, "diff", "--name-only", "--diff-filter=U").stdout.split() if f]
                if conflicted:
                    stats["merges_with_conflict"] += 1
                    for f in conflicted:
                        nh, nl = resolve(wt / f)
                        stats["conflict_hunks"] += nh
                        stats["conflict_lines"] += nl
                        stats["conflict_files"][f] = stats["conflict_files"].get(f, 0) + 1
                    _commit(wt, "merge main (resolved)", _date(day, d, 5, 2))
                elif r.returncode != 0:
                    raise RuntimeError(r.stderr)
                broken = gate(wt)
                if broken:
                    stats["broken_found"] += len(broken)
                    schema = (wt / "schema.txt").read_text().split()
                    for fid in broken:
                        p = wt / "features" / f"{fid}.txt"
                        col = p.read_text().split(":", 1)[1].strip()
                        while col not in schema and col in renamed_to:
                            col = renamed_to[col]
                        old = p.read_text().split(":", 1)[1].strip()
                        born = max(births.get(fid, 1), rename_day.get(old, 1))
                        stats["delays"].append(day - born)
                        p.write_text(f"uses: {col}\n")
                    _commit(wt, f"fix {len(broken)} features after merge", _date(day, d, 5, 3))
                    stats["fix_commits"] += 1
                git(main, "merge", "-q", "--ff-only", f"dev{d}")
            stats["main_red_after_round"] += bool(gate(main))
            for d, wt in enumerate(wts):
                git(wt, "merge", "-q", "--ff-only", "main")
            since_sync = [0] * N_DEVS
    stats["final_main_broken"] = len(gate(main))
    stats["mean_delay_days"] = (sum(stats["delays"]) / len(stats["delays"])) if stats["delays"] else 0.0
    stats["max_delay_days"] = max(stats["delays"]) if stats["delays"] else 0
    return stats
