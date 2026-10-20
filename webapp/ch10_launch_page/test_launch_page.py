from launch_page import audit

def test_complete_public_bundle_passes():
    pages = {k: {"url": f"https://card.example/{k}", "public": True, "app_name": "Card Five"} for k in ("landing", "privacy", "support", "account_deletion")}
    assert audit({"app_name": "Card Five", "pages": pages})["pass"]

def test_login_wall_or_name_drift_fails():
    pages = {k: {"url": f"https://card.example/{k}", "public": True, "app_name": "Card Five"} for k in ("landing", "privacy", "support", "account_deletion")}
    pages["privacy"]["public"] = False
    assert not audit({"app_name": "Card Five", "pages": pages})["pass"]
