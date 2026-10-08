"""3장 데이터 해시: 표의 '내용'이 같은지를 짧은 문자열 하나로 판정한다.

정규화 규칙
1. 열 이름을 정렬한다(열 순서만 바뀐 표는 같은 표로 본다).
2. 열마다 이름, dtype, 모양, 원시 바이트(little-endian, C 순서)를 차례로 넣는다.
   그래서 값 하나, dtype(float64 -> float32), 행 수가 바뀌면 해시가 바뀐다.
"""

import hashlib
import io

import numpy as np


def dataset_hash(table):
    h = hashlib.sha256()
    for name in sorted(table):
        a = np.ascontiguousarray(table[name])
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        h.update(name.encode())
        h.update(b"\x00" + a.dtype.str.encode() + b"\x00" + repr(a.shape).encode() + b"\x00")
        h.update(a.tobytes())
    return h.hexdigest()


def path_identity(path):
    """흔한 대용품: 파일 이름을 데이터의 이름으로 쓴다. 내용이 바뀌어도 그대로다."""
    return str(path)


def csv_roundtrip(table, digits=6):
    """표를 유효숫자 digits자리 CSV로 썼다가 다시 읽는다. 눈으로 보면 같은 표다."""
    names = list(table)
    buf = io.StringIO()
    buf.write(",".join(names) + "\n")
    n = len(table[names[0]])
    for i in range(n):
        buf.write(",".join(f"{table[k][i]:.{digits}g}" for k in names) + "\n")
    buf.seek(0)
    rows = [line.split(",") for line in buf.read().strip().split("\n")[1:]]
    out = {}
    for j, k in enumerate(names):
        col = [r[j] for r in rows]
        out[k] = np.array(col, dtype=table[k].dtype) if table[k].dtype.kind == "i" else np.array(col, dtype=float)
    return out, len(buf.getvalue().encode())
