"""명세 통합 문서 모형: raw(붙여 넣기만) -> clean(수식으로만 정리) -> pivot -> metrics, 그리고 checks.

specs/sheets_workbook_spec.md의 시트 구성을 파이썬 함수로 옮겼다. 표준 라이브러리만 쓴다.
pivot은 같은 계산을 세 방식(dict, sqlite3, NumPy)으로 구현해 대조한다.
"""

import hashlib
import re
import sqlite3
from collections import Counter, defaultdict

import productivity_ch05_data as DATA

COLS = ["booking_id", "date", "branch", "room", "hours", "status", "customer"]

# lookup 시트: 입력값 -> 표준값. 표준값 네 개 밖은 모두 '상태 모름'으로 뺀다.
STATUS_MAP = {
    "예약": "예약", "예약됨": "예약",
    "취소": "취소", "취소됨": "취소", "cancel": "취소",
    "노쇼": "노쇼", "no-show": "노쇼",
    "완료": "완료", "이용완료": "완료", "done": "완료",
}
YEAR = 2026


class ProtectedError(Exception):
    pass


class Raw:
    """raw_bookings 시트. 붙여 넣은 뒤에는 고칠 수 없다(시트 보호 'Restrict who can edit' 가정).

    source_rows는 내보낸 쪽이 함께 넘긴 행 수(파일 이름이나 첫 줄에 적어 두는 개수)다.
    """

    def __init__(self, rows, source_rows=None):
        self._rows = tuple(tuple(r[c] for c in COLS) for r in rows)
        self.source_rows = len(self._rows) if source_rows is None else source_rows
        self.fingerprint = self._hash()

    def _hash(self):
        h = hashlib.sha256()
        for r in self._rows:
            h.update("\x1f".join(r).encode())
        return h.hexdigest()

    def rows(self):
        return [dict(zip(COLS, r)) for r in self._rows]

    def __len__(self):
        return len(self._rows)

    def edit(self, *a, **k):
        raise ProtectedError("raw_bookings는 보호된 시트예요. 정렬·삭제는 clean이나 별도 시트에서 하세요.")


# ------------------------------------------------------------------ clean
def parse_date(s):
    """'2026-10-05' 또는 '10/5'(연도는 2026으로 둔다)만 받는다. '5일'처럼 달이 없으면 None."""
    s = s.strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})", s)
    if m:
        return f"{YEAR}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return None


def clean(raw, branches=None):
    """반환: (clean 행 목록, 뺀 행 목록[(행, 사유)]). 원본 행 하나는 정확히 둘 중 한 곳에 간다."""
    branches = set(DATA.BRANCHES if branches is None else branches)
    kept, dropped = [], []
    for r in raw.rows():
        d = parse_date(r["date"])
        st = STATUS_MAP.get(r["status"].strip())
        try:
            h = float(r["hours"])
        except ValueError:
            h = None
        if d is None:
            dropped.append((r, "날짜 모름"))
        elif st is None:
            dropped.append((r, "상태 모름"))
        elif r["branch"] not in branches:
            dropped.append((r, "지점 모름"))
        elif h is None or not (0 < h <= 12):
            dropped.append((r, "이용 시간 범위 밖"))
        else:
            kept.append({**r, "date": d, "status": st, "hours": h})
    return kept, dropped


# ------------------------------------------------------------------ pivot 세 가지 구현
def pivot_dict(rows):
    out = defaultdict(Counter)
    for r in rows:
        out[r["branch"]][r["status"]] += 1
    return {b: dict(c) for b, c in out.items()}


def pivot_sql(rows):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE c (branch TEXT, status TEXT)")
    con.executemany("INSERT INTO c VALUES (?, ?)", [(r["branch"], r["status"]) for r in rows])
    out = defaultdict(dict)
    for b, s, n in con.execute("SELECT branch, status, COUNT(*) FROM c GROUP BY branch, status"):
        out[b][s] = n
    con.close()
    return dict(out)


def pivot_numpy(rows):
    import numpy as np

    bs = sorted({r["branch"] for r in rows})
    ss = sorted({r["status"] for r in rows})
    bi = {b: i for i, b in enumerate(bs)}
    si = {s: i for i, s in enumerate(ss)}
    idx = np.array([bi[r["branch"]] * len(ss) + si[r["status"]] for r in rows], dtype=np.int64)
    grid = np.bincount(idx, minlength=len(bs) * len(ss)).reshape(len(bs), len(ss))
    return {b: {s: int(grid[i, j]) for j, s in enumerate(ss) if grid[i, j]} for b, i in bi.items()}


def pivot_total(p):
    return sum(sum(c.values()) for c in p.values())


# ------------------------------------------------------------------ metrics
def branch_rates(rows, branches=None):
    """지점별 취소율 = 취소 / 그 지점 전체(표준 상태 네 개). 분모가 0이면 None."""
    branches = DATA.BRANCHES if branches is None else branches
    p = pivot_dict(rows)
    out = {}
    for b in branches:
        c = p.get(b, {})
        n = sum(c.values())
        out[b] = c.get("취소", 0) / n if n else None
    return out


def metrics(rows, branches=None):
    rates = branch_rates(rows, branches)
    vals = [v for v in rates.values() if v is not None]
    n = len(rows)
    return {
        "mean_branch_cancel_rate": sum(vals) / len(vals) if vals else None,  # AVERAGE(clean 지점별 취소율)
        "pooled_cancel_rate": sum(r["status"] == "취소" for r in rows) / n if n else None,
        "noshow_rate": sum(r["status"] == "노쇼" for r in rows) / n if n else None,
        "n_branches_in_metric": len(vals),
        "zero_denominator_branches": [b for b, v in rates.items() if v is None],
    }


# ------------------------------------------------------------------ checks
DROP_REASON_CHECK = {"상태 모름": "C3", "날짜 모름": "C4", "지점 모름": "C7", "이용 시간 범위 밖": "C9"}


def checks(raw, branches=None, import_fingerprint=None):
    """checks 시트. 모두 True여야 한다. 반환: {이름: (통과 여부, 값)}.

    clean()이 행을 빼는 사유마다 그 개수가 0인지 보는 점검이 하나씩 있다(DROP_REASON_CHECK).
    사유는 있는데 점검이 없으면 C1은 맞는데도 행이 조용히 빠진다.
    """
    branches = DATA.BRANCHES if branches is None else branches
    kept, dropped = clean(raw, branches)
    p = pivot_dict(kept)
    reasons = Counter(why for _, why in dropped)
    m = metrics(kept, branches)
    out = {
        "C1 원본 행 = 정리 + 뺀 행": (len(raw) == len(kept) + len(dropped), [len(raw), len(kept), len(dropped)]),
        "C2 피벗 합계 = 정리 행": (pivot_total(p) == len(kept), [pivot_total(p), len(kept)]),
        "C3 표준 밖 상태 0": (reasons["상태 모름"] == 0, reasons["상태 모름"]),
        "C4 날짜 변환 실패 0": (reasons["날짜 모름"] == 0, reasons["날짜 모름"]),
        "C5 분모 0인 지점 0": (not m["zero_denominator_branches"], m["zero_denominator_branches"]),
        "C6 보낸 행 수 = 원본 행": (raw.source_rows == len(raw), [raw.source_rows, len(raw)]),
        "C7 목록 밖 지점 0": (reasons["지점 모름"] == 0, reasons["지점 모름"]),
        "C9 이용 시간 범위 밖 0": (reasons["이용 시간 범위 밖"] == 0, reasons["이용 시간 범위 밖"]),
    }
    if import_fingerprint is not None:
        out["C8 원본 지문 그대로"] = (raw.fingerprint == import_fingerprint, raw.fingerprint[:8])
    return out


def failed(ck):
    return [k for k, (ok, _) in ck.items() if not ok]
