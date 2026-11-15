"""자유 입력 시트 모형: 범위 수식, 행 추가, 자동 형식 변환, 행 수 상한이 있는 형식으로 저장.

실제 Google Sheets·Excel의 엔진이 아니라, 이 장의 사고를 재현하는 데 필요한 동작만 흉내 낸 모형이다.
모형의 가정(실제 도구 동작과 다를 수 있음)은 함수마다 적었다.

무엇을 재는가: (1) 닫힌 범위 수식(D2:D16)이 행을 맨 아래에 붙여도 늘지 않아 새 행을 빼먹는 사고, (2) 형식 자동 변환(2-03 -> 날짜 등),
(3) 행 수 상한이 있는 형식으로 저장할 때 뒤쪽 행이 말없이 잘리는 사고. 입력은 글자·행 목록이고 출력은 평가 결과나 잘린 목록이다.
테스트의 기대값(평균 10% 대 17%, 65,535행 보존 등)이다(장 본문 숫자와의 대응은 README로 확인 필요). 행 번호는 시트 기준 1부터(1행 = 머리글)이다.
"""

import re

# 상한 숫자의 근거는 아래 주석의 문서다(버전마다 다를 수 있어 확인 필요). 2의 16제곱
XLS_MAX_ROWS = 65_536  # Microsoft 지원 문서: Excel 97-2003은 65,536행 x 256열 (머리글 포함)


class Range:
    """'C2:C16'(닫힌 범위) 또는 'C2:C'(열 끝까지 열린 범위)."""

    def __init__(self, text):
        m = re.fullmatch(r"([A-Z])(\d+):([A-Z])(\d*)", text)
        if not m:
            raise ValueError(text)
        # 열은 한 글자(A~Z)만 지원한다. 끝 행 숫자가 비어 있으면(예: "C2:C") 열 끝까지 열린 범위
        self.col, self.start = m.group(1), int(m.group(2))
        self.end = int(m.group(4)) if m.group(4) else None

    def text(self):
        """범위를 다시 글자로(끝 행이 None이면 열린 범위로 "C2:C")."""
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
        self.rows = [list(r) for r in rows]  # 2행부터 (self.rows[0]이 시트의 2행)
        self.formulas = []  # (함수 이름, Range) 쌍들

    def n_rows(self):
        return len(self.rows) + 1  # 머리글 포함

    def column(self, col, rng):
        """범위에 든 칸의 값 목록. 열린 범위(end None)는 현재 마지막 행까지 읽는다. 시트 밖 행 번호는 건너뛴다."""
        j = self.cols.index(col)
        end = self.n_rows() if rng.end is None else rng.end
        return [self.rows[i - 2][j] for i in range(rng.start, end + 1) if 2 <= i <= self.n_rows()]

    def add_formula(self, func, rng_text):
        f = (func, Range(rng_text))
        self.formulas.append(f)
        return f

    def append_row(self, row):
        at = self.n_rows() + 1  # 새 행의 시트 행 번호. 닫힌 범위의 끝 + 1이면 범위는 늘지 않는다(on_insert 참고)
        for _, r in self.formulas:
            r.on_insert(at)
        self.rows.append(list(row))

    def insert_row(self, at, row):
        """at번째 행 자리에 끼워 넣는다(그 아래 행은 밀린다). 범위 수식 보정은 on_insert가 한다."""
        for _, r in self.formulas:
            r.on_insert(at)
        self.rows.insert(at - 2, list(row))

    def evaluate(self, f):
        """수식 f = (함수, 범위)의 값. 숫자 칸만 AVERAGE/SUM에 넣고(글자·빈칸·불리언 제외), COUNTA는 빈 칸이 아닌 칸을 센다."""
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
    # 달 1~12, 날 1~31만 보고 월별 일수는 따지지 않는다(예: "2-31"도 날짜로 바꾼다. 모형의 단순화)
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
    """열 전체를 붙여 넣는다. 반환: (종류, 값) 목록. plain_text=True는 붙여 넣기 전에 칸 형식을 '일반 텍스트'로 바꿔 둔 경우."""
    if plain_text:
        return [("text", v) for v in values]
    return [auto_convert(v) for v in values]


# ------------------------------------------------------------------ 행 수 상한이 있는 형식으로 저장
def save_capped(lines, max_rows=XLS_MAX_ROWS, header=True):
    """머리글 포함 max_rows 행까지만 남기고 나머지는 버린다. 오류를 내지 않는다(자동 처리 과정 가정).

    Excel 화면에서 .xls로 저장하면 호환성 검사가 경고하지만(Microsoft 지원 문서),
    사람 없이 도는 자동 변환에서는 그 경고를 볼 사람이 없다는 가정이다.
    """
    keep = max_rows - (1 if header else 0)  # 머리글이 1행을 차지하므로 데이터는 65,535행까지
    return list(lines[:keep])
