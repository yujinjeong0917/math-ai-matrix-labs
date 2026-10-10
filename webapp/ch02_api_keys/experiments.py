"""1인 개발 앱 출시 실전 2장 실험. `uv run python webapp/ch02_api_keys/experiments.py` 로 실행한다.

E0 키 구분표: 공개용·비밀·경우에 따라 다름이 몇 종인가
E1 1장 편집기 배포 폴더(public/)와 서버 폴더(server_only/)를 점검기로 훑기
E2 빌드 두 판(good, mistake): 브라우저 코드에 들어간 환경 변수 이름과 점검 결과
E3 RLS 모형: 같은 카드 12장을 키·로그인·정책에 따라 몇 행 읽나
E4 진짜 접두사 규칙 자체 시험: 실행 중에 만든 문자열로만(디스크에 남기지 않음)
E5 대조: 빌드 결과가 지금 원본과 같은가, SQL과 파이썬 필터가 같은 행 수를 내나

표준 라이브러리만 쓴다. 네트워크와 진짜 키 없이 돌고 결과는 매번 같다.
"""

import base64
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import webapp_ch02_build as B  # noqa: E402
import webapp_ch02_rls as R  # noqa: E402
import webapp_ch02_scan as S  # noqa: E402

OUT = HERE / "results" / "ch02.json"
CH01 = HERE.parent / "ch01_everything_visible" / "editor"


def synthetic_samples():
    """진짜 접두사 규칙을 시험할 문자열. 실행 중에만 만들고 파일에 쓰지 않아요."""
    rules = S.load_rules()["rules"]
    body = "x" * 24
    samples = []
    for r in rules:
        if r["prefix"] and r["id"] != "supabase_legacy_jwt":
            samples.append((r["id"], r["kind"], f'const k = "{r["prefix"]}{body}";'))

    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")

    for role, kind in (("anon", "public"), ("service_role", "secret")):
        tok = b64({"alg": "HS256", "typ": "JWT"}) + "." + b64({"iss": "test", "role": role}) + ".sig" + body
        samples.append(("supabase_legacy_jwt", kind, f'const k = "{tok}";'))
    # 경계: 다른 낱말 속에 접두사 글자가 섞인 경우는 찾지 않아야 해요
    near_miss = ['const risk_test_level = "aaaaaaaaaaaa";', 'const desk_live_mode = "bbbbbbbbbb";']
    return samples, near_miss


def e4_self_test():
    samples, near = synthetic_samples()
    hits = 0
    kinds_ok = 0
    for rid, kind, text in samples:
        f = [x for x in S.scan_text(text, "real") if x["rule"] == rid]
        hits += 1 if f else 0
        kinds_ok += 1 if f and f[0]["kind"] == kind else 0
    false_pos = sum(len([x for x in S.scan_text(t, "real") if not x["rule"].startswith("name:")]) for t in near)
    return {"samples": len(samples), "detected": hits, "kind_correct": kinds_ok,
            "near_miss_samples": len(near), "near_miss_flagged": false_pos}


def scan_both(path):
    out = {}
    for mode in ("real", "demo"):
        rep = S.scan_dir(path, mode)
        out[mode] = {"summary": S.summary(rep),
                     "findings": [{k: f[k] for k in ("file", "line", "rule", "kind", "shown")} for f in rep["findings"]]}
    return out


def main():
    rules = S.load_rules()["rules"]
    e0 = {"rules": len(rules),
          "by_kind": {k: sum(1 for r in rules if r["kind"] == k) for k in ("public", "secret", "by_role", "depends")},
          "with_documented_prefix": sum(1 for r in rules if r["prefix"]),
          "without_prefix": [r["id"] for r in rules if not r["prefix"]]}

    e1 = {"ch01_public": scan_both(CH01 / "public"), "ch01_server_only": scan_both(CH01 / "server_only")}

    e2 = {}
    for which in B.BUILDS:
        b = B.build(which, write=False)
        dist = B.PROJECT / B.BUILDS[which][2]
        rep = scan_both(dist)
        values_in_map = [k for k, v in b["env"].items() if v in b["map"]]
        e2[which] = {"inlined": b["inlined"], "env_names": list(b["env"]),
                     "not_in_browser": [k for k in b["env"] if k not in b["inlined"]],
                     "app_bytes": len(b["out"].encode()), "values_in_source_map": values_in_map,
                     "scan": rep}
    e2["env_files_in_project_source"] = scan_both(B.PROJECT)["demo"]["summary"]["env_files"]

    e3 = R.run_scenarios()

    fresh = all((B.PROJECT / B.BUILDS[w][2] / "app.js").read_text(encoding="utf-8") == B.build(w, write=False)["out"]
                for w in B.BUILDS)
    e5 = {"dist_matches_build": fresh,
          "sql_equals_python": all(r.get("rows") == r.get("python_rows") for r in e3 if r["rows"] is not None)}

    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "checked": "2026-10-10"},
        "e0_rules": e0, "e1_ch01": e1, "e2_builds": e2, "e3_rls": e3, "e4_real_rules_self_test": e4_self_test(),
        "e5_crosscheck": e5,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
