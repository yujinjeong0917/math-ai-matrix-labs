REQUIRED = ("identity", "signing", "version", "listing", "review_access", "privacy", "testing")

def audit(items):
    result = []
    for name in REQUIRED:
        row = items.get(name, {})
        status = row.get("status")
        evidence = str(row.get("evidence", "")).strip()
        result.append({"item": name, "pass": status == "verified" and bool(evidence)})
    return {"ready": all(r["pass"] for r in result), "items": result}
