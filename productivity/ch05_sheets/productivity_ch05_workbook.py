"""명세 통합 문서 모형: raw(붙여 넣기만) -> clean(수식으로만 정리) -> pivot -> metrics, 그리고 checks.

specs/sheets_workbook_spec.md의 시트 구성을 파이썬 함수로 옮겼다. 표준 라이브러리만 쓴다.
pivot은 같은 계산을 세 방식(dict, sqlite3, NumPy)으로 구현해 대조한다.

무엇을 재는가: 붙여 넣은 원본(raw)을 수식만으로 정리(clean)하며 "조용히 빠지는 행"을 세고, 지점별 취소율 지표와 점검표(checks, C1~C9)가
의도한 사고를 각각 잡는지 본다. 입력: 행 사전 목록(COLS의 7개 열, 값은 모두 글자). 출력: 정리한 행, 뺀 행과 사유, 점검 결과 {이름: (통과, 값)}.
pivot_numpy만 NumPy가 필요하고 나머지는 표준 라이브러리다. 점검은 "통과해도 값이 맞다는 보장은 아니다"(test_valid_but_wrong_value_slips_through).
"""

import hashlib
import re
import sqlite3
from collections import Counter, defaultdict

import productivity_ch05_data as DATA

COLS = ["booking_id", "date", "branch", "room", "hours", "status", "customer"]

# lookup 시트: 입력값 -> 표준값. 표준값 네 개 밖은 모두 '상태 모름'으로 뺀다.
# 키에는 앞뒤 공백을 뗀 뒤(strip) 찾는다. 대소문자는 구분한다("Cancel"은 모른다)
STATUS_MAP = {
    "예약": "예약", "예약됨": "예약",
    "취소": "취소", "취소됨": "취소", "cancel": "취소",
    "노쇼": "노쇼", "no-show": "노쇼",
    "완료": "완료", "이용완료": "완료", "done": "완료",
}
YEAR = 2026


class ProtectedError(Exception):
    """보호된 raw 시트를 고치려 할 때 내는 오류."""
    pass


class Raw:
    """raw_bookings 시트. 붙여 넣은 뒤에는 고칠 수 없다(시트 보호 'Restrict who can edit' 가정).

    source_rows는 내보낸 쪽이 함께 넘긴 행 수(파일 이름이나 첫 줄에 적어 두는 개수)다.
    """

    def __init__(self, rows, source_rows=None):
        # 튜플로 굳혀 두어 외부에서 바꿀 수 없게 한다(보호 시트의 모형)
        self._rows = tuple(tuple(r[c] for c in COLS) for r in rows)
        self.source_rows = len(self._rows) if source_rows is None else source_rows
        self.fingerprint = self._hash()

    def _hash(self):
        """붙여 넣은 직후의 SHA-256 지문. 칸 사이는 \x1f로 구분하지만 행 사이에는 구분자가 없다(경계가 모호한 값은 구분하지 못할 수 있음)."""
        h = hashlib.sha256()
        for r in self._rows:
            h.update("\x1f".join(r).encode())
        return h.hexdigest()

    def rows(self):
        """원본 행을 사전 목록으로(호출할 때마다 새 사전)."""
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
    # 판정 순서: 날짜 -> 상태 -> 지점 -> 이용 시간. 행 하나는 처음 걸린 사유 하나로만 기록된다(여러 사유가 겹쳐도 하나)
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
        elif h is None or not (0 < h <= 12):  # 12시간 상한은 모형의 선택(상식 범위)
            dropped.append((r, "이용 시간 범위 밖"))
        else:
            kept.append({**r, "date": d, "status": st, "hours": h})
    return kept, dropped


# ------------------------------------------------------------------ pivot 세 가지 구현
def pivot_dict(rows):
    """지점 x 상태 건수 표(사전). 세 구현 중 기준이 되는 가장 단순한 방식."""
    out = defaultdict(Counter)
    for r in rows:
        out[r["branch"]][r["status"]] += 1
    return {b: dict(c) for b, c in out.items()}


def pivot_sql(rows):
    """같은 표를 sqlite3의 GROUP BY로."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE c (branch TEXT, status TEXT)")
    con.executemany("INSERT INTO c VALUES (?, ?)", [(r["branch"], r["status"]) for r in rows])
    out = defaultdict(dict)
    for b, s, n in con.execute("SELECT branch, status, COUNT(*) FROM c GROUP BY branch, status"):
        out[b][s] = n
    con.close()
    return dict(out)


def pivot_numpy(rows):
    """같은 표를 NumPy bincount로(이 함수를 부를 때만 numpy를 import한다)."""
    import numpy as np

    bs = sorted({r["branch"] for r in rows})
    ss = sorted({r["status"] for r in rows})
    bi = {b: i for i, b in enumerate(bs)}
    si = {s: i for i, s in enumerate(ss)}
    # (지점 번호, 상태 번호)를 한 정수 지점 번호 x 상태 수 + 상태 번호로 펼쳐 bincount한 뒤 (지점, 상태) 격자로 되돌린다
    idx = np.array([bi[r["branch"]] * len(ss) + si[r["status"]] for r in rows], dtype=np.int64)
    grid = np.bincount(idx, minlength=len(bs) * len(ss)).reshape(len(bs), len(ss))
    return {b: {s: int(grid[i, j]) for j, s in enumerate(ss) if grid[i, j]} for b, i in bi.items()}


def pivot_total(p):
    """pivot 표의 건수 합계(점검 C2가 정리 행 수와 맞는지 본다)."""
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
    """지표 묶음. mean_branch_cancel_rate는 지점별 취소율의 단순 평균(지점 크기를 가중하지 않음), pooled_cancel_rate는 전체 건수 기준이다.
    예약이 0건인 지점은 평균에서 빠진다(n_branches_in_metric이 그 수를 보여 준다)."""
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
# clean()이 행을 빼는 사유 -> 그 사유가 0이어야 통과하는 점검 번호. 사유가 늘면 이 표에도 점검을 더해야 한다(테스트가 둘의 짝을 확인)
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
    """checks 결과에서 통과하지 못한 점검 이름 목록."""
    return [k for k, (ok, _) in ck.items() if not ok]
