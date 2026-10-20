from submission import REQUIRED, audit

def test_evidence_is_required():
    items = {k: {"status": "verified", "evidence": f"release/{k}.md"} for k in REQUIRED}
    assert audit(items)["ready"]
    items["privacy"]["evidence"] = ""
    assert not audit(items)["ready"]

def test_todo_is_not_ready():
    items = {k: {"status": "verified", "evidence": "ok"} for k in REQUIRED}
    items["testing"]["status"] = "todo"
    assert not audit(items)["ready"]
