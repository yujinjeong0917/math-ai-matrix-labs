from events import completion_funnel, validate

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
