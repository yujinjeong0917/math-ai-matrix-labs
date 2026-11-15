"""제품 분석 이벤트 계약과 완료 퍼널 (1인 개발 앱 출시 실전 12장).

validate: 이벤트 1건이 계약(이름·속성)을 지키는지 판정해 (통과 여부, 사유)를 돌려준다.
completion_funnel: 이벤트 목록에서 4단계별 고유 사용자 수를 센다.
입력 이벤트 형식: {"user_id", "name", "properties"}
"""

# 이벤트 계약: 이벤트 이름 -> 허용하는 속성 이름 집합. 여기 없는 이름·속성은 거부한다
CONTRACT = {
    "project_start": {"entry_route", "template_id"},
    "asset_add": {"asset_type"},
    "export_complete": {"format", "page_count", "elapsed_ms"},
    "next_action_click": {"action"},
}
# 개인정보로 이어질 수 있는 속성. 계약에 이름이 있든 없든 먼저 걸러낸다
FORBIDDEN = {"email", "phone", "raw_text", "filename"}

def validate(event):
    name, props = event.get("name"), set(event.get("properties", {}))
    if name not in CONTRACT:
        return False, "unknown_event"
    # 금지 속성 검사가 허용 속성 검사보다 앞이라, 금지 속성은 항상 forbidden_data로 보고된다
    if props & FORBIDDEN:
        return False, "forbidden_data"
    # 부분집합이면 통과: 속성을 생략하는 것은 허용, 계약에 없는 속성만 거부
    if not props <= CONTRACT[name]:
        return False, "unknown_property"
    return True, "ok"

def completion_funnel(events):
    stages = ("project_start", "asset_add", "export_complete", "next_action_click")
    # 단계별로 사용자 id 집합을 모아, 같은 사람이 여러 번 해도 한 명으로 센다(이벤트 수가 아닌 사용자 수)
    users = {s: set() for s in stages}
    for event in events:
        ok, _ = validate(event)
        if ok:  # 계약 위반 이벤트는 퍼널에서 제외한다
            users[event["name"]].add(event["user_id"])
    return {s: len(users[s]) for s in stages}
