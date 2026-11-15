"""모델 파일(S3)과 컨테이너 이미지(ECR)의 버전 기록이 같은 릴리스를 가리키는지 검사한다.

AWS 호출 없음. 릴리스 기록 JSON만 읽는다. 값은 가상(예시용)이다.

재는 것: (1) 릴리스 기록의 규칙 R1~R7 위반, (2) 버전을 쌓을 때 저장량, (3) S3 멀티파트 업로드 파트 수.
입력: release_manifest.json. 출력: 규칙별 FAIL/WARN, 판정(BLOCK/WARN/PASS), 저장량 46.83GB,
100 MiB 파트 45개, 최소 파트 5 MiB에 894개 (README 기대 출력과 대응).
규칙 7개는 학습용이며 AWS 공식 요구사항이 아니다.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

MIB = 1024 * 1024
GIB = 1024**3
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
S3_MAX_PARTS = 10_000
# 파트 크기 범위는 아래 주석의 공식 문서 값. 파트 수 상한 10,000도 같은 문서 기준
S3_MIN_PART = 5 * MIB          # 공식 문서: 파트 크기 5 MiB ~ 5 GiB
S3_MAX_PART = 5 * GIB


def stored_bytes(size_bytes: int, versions: int) -> int:
    """버전마다 객체 전체가 따로 저장된다(차이분만이 아님): 크기 x 버전 수."""
    if size_bytes < 0 or versions < 1:
        raise ValueError("크기와 버전 수를 확인하세요")
    return size_bytes * versions


def multipart_plan(size_bytes: int, part_bytes: int) -> dict:
    """파트 크기로 나눈 파트 수와 10,000개 상한 안인지(fits)를 돌려준다."""
    if not (S3_MIN_PART <= part_bytes <= S3_MAX_PART):
        raise ValueError("파트 크기는 5 MiB ~ 5 GiB")
    parts = math.ceil(size_bytes / part_bytes)
    # 올림: 4,683,000,000 / 100 MiB = 44.66 -> 45파트. 마지막 파트는 더 작아도 된다
    return {"parts": parts, "fits": parts <= S3_MAX_PARTS}


def min_part_bytes(size_bytes: int) -> int:
    """10,000개 파트 안에 들어가는 가장 작은 파트 크기(5 MiB 아래로는 못 내려감)."""
    # 4.68GB를 10,000로 나누면 약 468KB로 5 MiB 하한보다 작아 하한이 적용된다(파트 894개)
    return max(S3_MIN_PART, math.ceil(size_bytes / S3_MAX_PARTS))


def check_release(m: dict) -> list[dict]:
    """위반 목록. 각 항목은 규칙 번호·심각도·이유."""
    found = []

    def add(rule: str, level: str, why: str) -> None:
        found.append({"rule": rule, "level": level, "why": why})

    # 값이 비었거나 키가 없으면 모두 위반으로 처리하도록 get 기본값을 쓴다
    model = m.get("model", {})
    if not model.get("s3_version_id"):
        add("R1", "FAIL", "모델 파일에 S3 버전 ID가 없어요")
    if not SHA256.match(model.get("sha256", "")):
        add("R2", "FAIL", "모델 파일 sha256이 64자리 16진수가 아니에요")
    ref = m.get("image", {}).get("reference", "")
    if not DIGEST.search(ref):
        add("R3", "FAIL", "이미지를 digest(@sha256:...)로 고정하지 않았어요")
    if ref.endswith(":latest"):
        add("R4", "FAIL", "latest 태그는 나중에 다른 이미지를 가리킬 수 있어요")
    # 기본값 MUTABLE은 이 코드의 선택(ECR 기본값 확인 아님). digest 고정(R3)이 있으면 위험이 작다고 보아 WARN
    if m.get("repository", {}).get("tag_mutability", "MUTABLE") != "IMMUTABLE":
        add("R5", "WARN", "ECR 태그 덮어쓰기를 막지 않았어요")
    if m.get("bucket", {}).get("versioning") != "Enabled":
        add("R6", "FAIL", "S3 버킷 버전 관리가 켜져 있지 않아요")
    if not m.get("code_commit"):
        add("R7", "FAIL", "서빙 코드 커밋이 기록돼 있지 않아요")
    return found


def verdict(m: dict) -> str:
    """FAIL이 하나라도 있으면 BLOCK, WARN만 있으면 WARN, 없으면 PASS."""
    f = check_release(m)
    if any(x["level"] == "FAIL" for x in f):
        return "BLOCK"
    return "WARN" if f else "PASS"


def main(path: str) -> None:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    for x in check_release(m):
        print(f"{x['rule']} {x['level']}: {x['why']}")
    print("판정:", verdict(m))
    size = m["model"]["size_bytes"]
    # GB는 1e9(십진). 4.683GB x 10 = 46.83GB
    print(f"버전 10개 보관 시 저장량: {stored_bytes(size, 10) / 1e9:.2f}GB")
    print(f"100 MiB 파트: {multipart_plan(size, 100 * MIB)}")
    p = min_part_bytes(size)
    print(f"파트 최소 크기: {p}바이트 = {p / MIB:.0f} MiB, 파트 수 {multipart_plan(size, p)['parts']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("release_manifest.json")))
