REQUIRED = ("landing", "privacy", "support", "account_deletion")

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
