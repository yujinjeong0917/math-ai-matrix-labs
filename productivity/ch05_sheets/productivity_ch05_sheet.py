"""자유 입력 시트 모형: 범위 수식, 행 추가, 자동 형식 변환, 행 수 상한이 있는 형식으로 저장.

실제 Google Sheets·Excel의 엔진이 아니라, 이 장의 사고를 재현하는 데 필요한 동작만 흉내 낸 모형이다.
모형의 가정(실제 도구 동작과 다를 수 있음)은 함수마다 적었다.
"""

import re

XLS_MAX_ROWS = 65_536  # Microsoft 지원 문서: Excel 97-2003은 65,536행 x 256열 (머리글 포함)


class Range:
    """'C2:C16'(닫힌 범위) 또는 'C2:C'(열 끝까지 열린 범위)."""

    def __init__(self, text):
        m = re.fullmatch(r"([A-Z])(\d+):([A-Z])(\d*)", text)
        if not m:
            raise ValueError(text)
        self.col, self.start = m.group(1), int(m.group(2))
        self.end = int(m.group(4)) if m.group(4) else None

    def text(self):
        return f"{self.col}{self.start}:{self.col}{'' if self.end is None else self.end}"

    def on_insert(self, at):
        """at번째 행 앞에 행을 끼워 넣을 때. 모형 가정: 범위 안쪽(start < at <= end)에 끼우면 범위가 늘고,
        범위 바로 아래(at = end + 1, 즉 맨 아래에 붙이기)는 늘지 않는다."""
        if self.end is None:
            return
        if at <= self.start:
            self.start += 1
            self.end += 1
        elif at <= self.end:
            self.end += 1


class Sheet:
    """열 이름(A, B, C...) -> 값 목록. 1행은 머리글이다."""

    def __init__(self, header, rows):
        self.cols = [chr(ord("A") + i) for i in range(len(header))]
        self.header = list(header)
        self.rows = [list(r) for r in rows]  # 2행부터
        self.formulas = []

    def n_rows(self):
        return len(self.rows) + 1  # 머리글 포함

    def column(self, col, rng):
        j = self.cols.index(col)
        end = self.n_rows() if rng.end is None else rng.end
        return [self.rows[i - 2][j] for i in range(rng.start, end + 1) if 2 <= i <= self.n_rows()]

    def add_formula(self, func, rng_text):
        f = (func, Range(rng_text))
        self.formulas.append(f)
        return f

    def append_row(self, row):
        at = self.n_rows() + 1
        for _, r in self.formulas:
            r.on_insert(at)
        self.rows.append(list(row))

    def insert_row(self, at, row):
        for _, r in self.formulas:
            r.on_insert(at)
        self.rows.insert(at - 2, list(row))

    def evaluate(self, f):
        func, rng = f
        vals = self.column(rng.col, rng)
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if func == "AVERAGE":  # 모형 가정: 빈칸과 글자는 건너뛴다
            return sum(nums) / len(nums) if nums else None
        if func == "SUM":
            return sum(nums)
        if func == "COUNTA":
            return sum(v not in (None, "") for v in vals)
        raise ValueError(func)


# ------------------------------------------------------------------ 자동 형식 변환(단순화한 흉내)
def auto_convert(text, year=2026):
    """칸에 글자를 붙여 넣을 때 도구가 '알아서' 바꾸는 동작을 단순화한 모형.

    - 'M-D' 또는 'M/D' 꼴이고 달·날이 말이 되면 날짜로 바꾼다(예: '2-03' -> 2026-02-03).
    - '숫자E숫자' 꼴이면 지수 표기 숫자로 바꾼다(Ziemann 등 2016이 든 RIKEN 번호의 예).
    - 앞자리 0이 있는 숫자는 숫자로 바꿔 0이 사라진다(예: '007' -> 7).
    실제 도구의 규칙은 언어·지역 설정과 버전에 따라 다르다. 여기서는 사고의 모양만 본다.
    """
    s = text.strip()
    m = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})", s)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return ("date", f"{year}-{mo:02d}-{d:02d}")
    if re.fullmatch(r"\d+E\d+", s):
        return ("number", float(s))
    if re.fullmatch(r"0\d+", s):
        return ("number", int(s))
    return ("text", s)


def paste_column(values, plain_text=False):
    """plain_text=True는 붙여 넣기 전에 칸 형식을 '일반 텍스트'로 바꿔 둔 경우."""
    if plain_text:
        return [("text", v) for v in values]
    return [auto_convert(v) for v in values]


# ------------------------------------------------------------------ 행 수 상한이 있는 형식으로 저장
def save_capped(lines, max_rows=XLS_MAX_ROWS, header=True):
    """머리글 포함 max_rows 행까지만 남기고 나머지는 버린다. 오류를 내지 않는다(자동 처리 과정 가정).

    Excel 화면에서 .xls로 저장하면 호환성 검사가 경고하지만(Microsoft 지원 문서),
    사람 없이 도는 자동 변환에서는 그 경고를 볼 사람이 없다는 가정이다.
    """
    keep = max_rows - (1 if header else 0)
    return list(lines[:keep])
