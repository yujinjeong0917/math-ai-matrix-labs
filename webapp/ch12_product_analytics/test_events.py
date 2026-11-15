"""events 모듈 테스트. 확인하는 것: 개인정보 속성 거부, 퍼널이 이벤트 수가 아닌 고유 사용자 수를 센다."""

from events import completion_funnel, validate

# 이메일 속성이 있으면 거부되고 사유가 forbidden_data여야 한다
def test_private_data_is_rejected():
    ok, reason = validate({"name": "project_start", "properties": {"email": "a@example.com"}})
    assert not ok and reason == "forbidden_data"

# u1이 project_start를 두 번 보내도 1명으로 센다. 기대값: start 2명(u1,u2), asset_add 1명(u1)
def test_funnel_counts_unique_users():
    events = [
        {"user_id": "u1", "name": "project_start", "properties": {}},
        {"user_id": "u1", "name": "project_start", "properties": {}},
        {"user_id": "u1", "name": "asset_add", "properties": {"asset_type": "photo"}},
        {"user_id": "u2", "name": "project_start", "properties": {}},
    ]
    assert completion_funnel(events)["project_start"] == 2
    assert completion_funnel(events)["asset_add"] == 1
