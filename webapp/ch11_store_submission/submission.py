"""스토어 제출 점검표 (1인 개발 앱 출시 실전 11장).

무엇을 재는지: 7개 항목이 모두 "verified"이고 근거가 비어 있지 않은지.
입력: {항목 이름: {"status": ..., "evidence": ...}}
출력: {"ready": 전체 통과 여부, "items": [{"item", "pass"}, ...]}
"""

# 제출 전 점검 항목 7개. 항목마다 상태(verified)와 근거(파일 경로 등)가 함께 있어야 통과(모형의 선택)
REQUIRED = ("identity", "signing", "version", "listing", "review_access", "privacy", "testing")

def audit(items):
    result = []
    for name in REQUIRED:
        row = items.get(name, {})
        status = row.get("status")
        # 근거는 문자열로 바꾼 뒤 공백을 걷어, 빈 값이나 공백만 있는 값을 거른다
        evidence = str(row.get("evidence", "")).strip()
        result.append({"item": name, "pass": status == "verified" and bool(evidence)})
    return {"ready": all(r["pass"] for r in result), "items": result}
