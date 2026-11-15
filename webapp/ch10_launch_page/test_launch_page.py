"""launch_page.audit 테스트. 확인하는 것: 공개 4페이지 완비 시 통과, 하나라도 비공개면 실패."""

from launch_page import audit

# 4개 페이지가 모두 https, 공개, 같은 앱 이름이면 통과한다
def test_complete_public_bundle_passes():
    pages = {k: {"url": f"https://card.example/{k}", "public": True, "app_name": "Card Five"} for k in ("landing", "privacy", "support", "account_deletion")}
    assert audit({"app_name": "Card Five", "pages": pages})["pass"]

# 개인정보 페이지를 비공개(로그인 벽)로 바꾸면 전체가 실패해야 한다.
# (이름 불일치 경로는 이 테스트에서 따로 확인하지 않는다)
def test_login_wall_or_name_drift_fails():
    pages = {k: {"url": f"https://card.example/{k}", "public": True, "app_name": "Card Five"} for k in ("landing", "privacy", "support", "account_deletion")}
    pages["privacy"]["public"] = False
    assert not audit({"app_name": "Card Five", "pages": pages})["pass"]
