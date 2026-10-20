CONTRACT = {
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
