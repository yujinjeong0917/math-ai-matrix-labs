"""출시 페이지 점검 (1인 개발 앱 출시 실전 10장).

무엇을 재는지: 랜딩, 개인정보처리방침, 지원, 계정 삭제 4개 페이지가 심사 전에 갖춰졌는지.
입력: snapshot dict. {"app_name": 앱 이름, "pages": {페이지 키: {"url", "public", "app_name"}}}
출력: {"pass": 전체 통과 여부, "rows": [{"page", "pass"}, ...]}
"""

# 스토어 심사에서 열어 보는 공개 페이지 4종. 하나라도 비공개면 심사에서 막힐 수 있다(모형의 선택).
REQUIRED = ("landing", "privacy", "support", "account_deletion")

def audit(snapshot):
    app = snapshot.get("app_name", "").strip()
    rows = []
    for key in REQUIRED:
        page = snapshot.get("pages", {}).get(key, {})
        url = page.get("url", "")
        public = bool(page.get("public"))
        # 앱 이름이 페이지마다 다르면 심사자가 같은 앱으로 보지 않을 수 있어 일치를 요구한다
        same_name = page.get("app_name") == app
        # 통과 조건: https 주소 + 로그인 없이 공개 + 앱 이름 일치
        rows.append({"page": key, "pass": url.startswith("https://") and public and same_name})
    # 앱 이름이 비어 있으면 이름 일치 검사가 무의미하므로 전체 실패로 본다
    return {"pass": bool(app) and all(r["pass"] for r in rows), "rows": rows}
