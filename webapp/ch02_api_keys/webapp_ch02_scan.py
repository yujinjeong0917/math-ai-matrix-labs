"""1인 개발 앱 출시 실전 2장: 배포 폴더에서 키 모양을 찾는 점검기.

규칙은 data/key_rules_2026-10-10.json에서 읽어요. 세 가지로 찾아요.
1. 접두사: 회사 공식 문서에 적힌 접두사(sb_secret_, pk_live_ 등) 뒤에 글자가 8개 이상 이어지면 찾아요.
2. 옛 Supabase 키(JWT): eyJ로 시작하는 점 세 토막 문자열의 가운데 토막을 풀어 role을 읽어요.
3. 이름: 모양을 모르는 키는 secret, service_role, api_key, private_key가 들어간 이름에 값이 붙어 있으면 찾아요.

mode="demo"이면 접두사 대신 실습용 표식(DEMO_SUPABASE_SECRET_ 등)을 찾아요. 실습 폴더의 가짜 값은
모두 이 표식으로만 만들었어요. 진짜 접두사 규칙은 테스트에서 실행 중에 만든 문자열로만 시험해요.

찾은 값은 그대로 적지 않아요. Supabase 문서의 권고대로 접두사 뒤 6글자까지만 보이고,
어느 키였는지는 SHA-256 해시 앞 16글자로 남겨요.
"""

import base64
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
RULES_FILE = HERE / "data" / "key_rules_2026-10-10.json"
TEXT_SUFFIXES = {".html", ".js", ".mjs", ".css", ".json", ".map", ".txt", ".svg", ".env", ".sql", ""}
BOUNDARY = r"(?<![A-Za-z0-9_\-])"
TAIL = r"[A-Za-z0-9_\-]{8,}"
JWT = re.compile(BOUNDARY + r"(eyJ[A-Za-z0-9_\-]+)\.([A-Za-z0-9_\-]+)\.([A-Za-z0-9_\-]+)")


def load_rules():
    return json.loads(RULES_FILE.read_text(encoding="utf-8"))


def mask(value, prefix):
    """접두사 + 그 뒤 최대 6글자 + '…'. 해시는 따로."""
    rest = value[len(prefix):]
    return prefix + rest[:6] + ("…" if len(rest) > 6 else "")


def fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def jwt_role(header_b64, payload_b64):
    try:
        pad = "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + pad))
        return payload.get("role")
    except Exception:
        return None


def scan_text(text, mode="real", rules=None):
    """문자열 하나에서 찾은 것 목록. 각 항목: rule, kind, line, shown, fp."""
    rules = rules or load_rules()
    found = []

    def add(rule_id, kind, pos, value, shown_prefix):
        line = text.count("\n", 0, pos) + 1
        found.append({"rule": rule_id, "kind": kind, "line": line,
                      "shown": mask(value, shown_prefix), "fp": fingerprint(value)})

    spans = []
    for r in rules["rules"]:
        if mode == "real":
            if not r["prefix"] or r["id"] == "supabase_legacy_jwt":
                continue
            start = r["prefix"]
        else:
            if not r["demo_marker"]:
                continue
            start = r["demo_marker"]
        pat = re.compile(BOUNDARY + re.escape(start) + (TAIL if mode == "real" else r"[A-Za-z0-9_\-]{2,}"))
        for m in pat.finditer(text):
            kind = r["kind"]
            if r["id"] == "supabase_legacy_jwt":   # 실습 표식: DEMO_LEGACY_JWT_ANON_.. / _SERVICE_ROLE_..
                kind = "secret" if "SERVICE_ROLE" in m.group(0) else "public"
            add(r["id"], kind, m.start(), m.group(0), start)
            spans.append((m.start(), m.end()))

    if mode == "real":
        for m in JWT.finditer(text):
            role = jwt_role(m.group(1), m.group(2))
            if role in ("anon", "service_role"):
                add("supabase_legacy_jwt", "secret" if role == "service_role" else "public",
                    m.start(), m.group(0), "eyJ")
                spans.append((m.start(), m.end()))

    for m in re.finditer(rules["name_rule"]["pattern"], text):
        v0 = m.start(2)
        if any(a <= v0 < b for a, b in spans):      # 이미 모양으로 찾은 값이면 건너뛰어요
            continue
        found.append({"rule": "name:" + m.group(1), "kind": "check", "line": text.count("\n", 0, m.start()) + 1,
                      "shown": m.group(2)[:6] + "…", "fp": fingerprint(m.group(2))})
    return found


def scan_dir(root, mode="real", rules=None):
    """폴더 아래 글자 파일을 모두 훑어요. .env 같은 설정 파일이 배포 폴더에 있는지도 봐요."""
    rules = rules or load_rules()
    root = Path(root)
    report = {"files": 0, "bytes": 0, "findings": [], "env_files": []}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        if p.name.startswith(".env") or p.name.startswith("env."):
            report["env_files"].append(rel)
        if p.suffix not in TEXT_SUFFIXES and not p.name.startswith(("env.", ".env")):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        report["files"] += 1
        report["bytes"] += p.stat().st_size
        for f in scan_text(text, mode, rules):
            f["file"] = rel
            report["findings"].append(f)
    return report


def summary(report):
    out = {"public": 0, "secret": 0, "depends": 0, "check": 0}
    for f in report["findings"]:
        out[f["kind"]] += 1
    out["files"] = report["files"]
    out["env_files"] = len(report["env_files"])
    out["pass"] = out["secret"] == 0 and out["env_files"] == 0
    return out


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "project" / "dist_mistake")
    mode = sys.argv[2] if len(sys.argv) > 2 else "demo"
    rep = scan_dir(target, mode)
    for f in rep["findings"]:
        print(f"{f['kind']:7} {f['file']}:{f['line']}  {f['rule']}  {f['shown']}  sha256:{f['fp']}")
    s = summary(rep)
    print(s)
    raise SystemExit(0 if s["pass"] else 1)
