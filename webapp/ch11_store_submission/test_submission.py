"""submission.audit 테스트. 확인하는 것: 근거 필수, verified 외 상태는 불합격."""

from submission import REQUIRED, audit

# 근거가 있으면 통과, 근거를 빈 문자열로 바꾸면 실패해야 한다
def test_evidence_is_required():
    items = {k: {"status": "verified", "evidence": f"release/{k}.md"} for k in REQUIRED}
    assert audit(items)["ready"]
    items["privacy"]["evidence"] = ""
    assert not audit(items)["ready"]

# 상태가 verified가 아니면(todo) 근거가 있어도 준비 안 됨
def test_todo_is_not_ready():
    items = {k: {"status": "verified", "evidence": "ok"} for k in REQUIRED}
    items["testing"]["status"] = "todo"
    assert not audit(items)["ready"]
