"""웹앱 트랙 10~13장 실습을 결정적으로 생성한다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = {
"webapp/ch10_launch_page/README.md": """# 10장. 소개 페이지 출시 점검

웹 챕터: https://math-ai-matrix-pi.vercel.app/tracks/webapp/10-launch-page.html

소개·개인정보·지원·계정 삭제 URL이 공개돼 있고 같은 앱 이름을 쓰는지 검사합니다. 네트워크를 호출하지 않고 제출용 스냅샷 사전을 판정합니다.

```bash
uv run pytest -q webapp/ch10_launch_page
```
""",
"webapp/ch10_launch_page/launch_page.py": '''REQUIRED = ("landing", "privacy", "support", "account_deletion")

def audit(snapshot):
    app = snapshot.get("app_name", "").strip()
    rows = []
    for key in REQUIRED:
        page = snapshot.get("pages", {}).get(key, {})
        url = page.get("url", "")
        public = bool(page.get("public"))
        same_name = page.get("app_name") == app
        rows.append({"page": key, "pass": url.startswith("https://") and public and same_name})
    return {"pass": bool(app) and all(r["pass"] for r in rows), "rows": rows}
''',
"webapp/ch10_launch_page/test_launch_page.py": '''from launch_page import audit

def test_complete_public_bundle_passes():
    pages = {k: {"url": f"https://card.example/{k}", "public": True, "app_name": "Card Five"} for k in ("landing", "privacy", "support", "account_deletion")}
    assert audit({"app_name": "Card Five", "pages": pages})["pass"]

def test_login_wall_or_name_drift_fails():
    pages = {k: {"url": f"https://card.example/{k}", "public": True, "app_name": "Card Five"} for k in ("landing", "privacy", "support", "account_deletion")}
    pages["privacy"]["public"] = False
    assert not audit({"app_name": "Card Five", "pages": pages})["pass"]
''',
"webapp/ch11_store_submission/README.md": """# 11장. 스토어 제출 묶음 점검

웹 챕터: https://math-ai-matrix-pi.vercel.app/tracks/webapp/11-store-submission.html

항목을 체크했다는 말 대신 근거 URL·파일 위치가 있는지를 확인합니다. 정책 값은 저장하지 않고 제출 당일 공식 콘솔에서 사람이 다시 확인할 항목으로 남깁니다.
""",
"webapp/ch11_store_submission/submission.py": '''REQUIRED = ("identity", "signing", "version", "listing", "review_access", "privacy", "testing")

def audit(items):
    result = []
    for name in REQUIRED:
        row = items.get(name, {})
        status = row.get("status")
        evidence = str(row.get("evidence", "")).strip()
        result.append({"item": name, "pass": status == "verified" and bool(evidence)})
    return {"ready": all(r["pass"] for r in result), "items": result}
''',
"webapp/ch11_store_submission/test_submission.py": '''from submission import REQUIRED, audit

def test_evidence_is_required():
    items = {k: {"status": "verified", "evidence": f"release/{k}.md"} for k in REQUIRED}
    assert audit(items)["ready"]
    items["privacy"]["evidence"] = ""
    assert not audit(items)["ready"]

def test_todo_is_not_ready():
    items = {k: {"status": "verified", "evidence": "ok"} for k in REQUIRED}
    items["testing"]["status"] = "todo"
    assert not audit(items)["ready"]
''',
"webapp/ch12_product_analytics/README.md": """# 12장. 제품 분석 이벤트 계약

웹 챕터: https://math-ai-matrix-pi.vercel.app/tracks/webapp/12-product-analytics.html

이벤트 이름·발생 조건·허용 속성을 계약으로 만들고, 중복 완료와 금지 개인정보를 찾습니다.
""",
"webapp/ch12_product_analytics/events.py": '''CONTRACT = {
    "project_start": {"entry_route", "template_id"},
    "asset_add": {"asset_type"},
    "export_complete": {"format", "page_count", "elapsed_ms"},
    "next_action_click": {"action"},
}
FORBIDDEN = {"email", "phone", "raw_text", "filename"}

def validate(event):
    name, props = event.get("name"), set(event.get("properties", {}))
    if name not in CONTRACT:
        return False, "unknown_event"
    if props & FORBIDDEN:
        return False, "forbidden_data"
    if not props <= CONTRACT[name]:
        return False, "unknown_property"
    return True, "ok"

def completion_funnel(events):
    stages = ("project_start", "asset_add", "export_complete", "next_action_click")
    users = {s: set() for s in stages}
    for event in events:
        ok, _ = validate(event)
        if ok:
            users[event["name"]].add(event["user_id"])
    return {s: len(users[s]) for s in stages}
''',
"webapp/ch12_product_analytics/test_events.py": '''from events import completion_funnel, validate

def test_private_data_is_rejected():
    ok, reason = validate({"name": "project_start", "properties": {"email": "a@example.com"}})
    assert not ok and reason == "forbidden_data"

def test_funnel_counts_unique_users():
    events = [
        {"user_id": "u1", "name": "project_start", "properties": {}},
        {"user_id": "u1", "name": "project_start", "properties": {}},
        {"user_id": "u1", "name": "asset_add", "properties": {"asset_type": "photo"}},
        {"user_id": "u2", "name": "project_start", "properties": {}},
    ]
    assert completion_funnel(events)["project_start"] == 2
    assert completion_funnel(events)["asset_add"] == 1
''',
"webapp/ch13_release_learning/README.md": """# 13장. 출시 뒤 다음 실험 고르기

웹 챕터: https://math-ai-matrix-pi.vercel.app/tracks/webapp/13-release-learning.html

단계별 고유 사용자 수로 퍼널을 계산하고 가장 큰 직전 단계 이탈을 찾습니다. 이 결과는 원인 판정이 아니라 다음 조사 후보입니다.
""",
"webapp/ch13_release_learning/funnel.py": '''def analyze(counts):
    if not counts or any(v < 0 for _, v in counts) or any(counts[i][1] < counts[i + 1][1] for i in range(len(counts) - 1)):
        raise ValueError("counts must be non-negative and non-increasing")
    rows, worst = [], None
    for i, (stage, count) in enumerate(counts):
        rate = None if i == 0 or counts[i - 1][1] == 0 else count / counts[i - 1][1]
        row = {"stage": stage, "count": count, "step_rate": rate}
        rows.append(row)
        if rate is not None and (worst is None or rate < worst["step_rate"]):
            worst = row
    return {"rows": rows, "investigate_first": None if worst is None else worst["stage"]}
''',
"webapp/ch13_release_learning/test_funnel.py": '''import pytest
from funnel import analyze

def test_finds_largest_step_drop():
    result = analyze([("install", 1000), ("open", 600), ("start", 300), ("input", 90), ("export", 54)])
    assert result["investigate_first"] == "input"
    assert result["rows"][3]["step_rate"] == pytest.approx(.3)

def test_rejects_mixed_or_broken_counts():
    with pytest.raises(ValueError):
        analyze([("start", 10), ("finish", 12)])
''',
}

for rel, body in FILES.items():
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
print(f"generated {len(FILES)} lab files")
