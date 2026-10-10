"""1인 개발 앱 출시 실전 6장 실험. `uv run python webapp/ch06_app_packaging/experiments.py` 로 실행한다.

E0 PWA로 바꾸려고 더한 파일: index.html 에 더한 줄, manifest.json, sw.js, 아이콘 두 개의 크기
E1 manifest 검사기: 우리 manifest와 일부러 망가뜨린 4개
E2 precache 목록과 크기(1장의 첫 화면 7,659바이트와 비교)
E3 첫 방문 요청 손계산: 페이지 GET 8 + AI 버튼 POST 1 + precache 12
E4 오프라인 캐시 전략 모형: 서비스 워커 없음 / cache-first / network-first / stale-while-revalidate / 섞어 쓰기(sw.js 규칙),
   배포 때 sw.js 버전을 바꿨을 때와 안 바꿨을 때
E5 네 가지 길 비교표와 조건별 고르기
E6 대조: sw.js 의 precache 파일이 모두 있는가, 서버가 보낸 크기 = 디스크 크기, node --check

표준 라이브러리만 쓴다. 서버는 127.0.0.1의 빈 포트에 잠깐 띄웠다 끈다. 결과는 매번 같다.
"""

import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import webapp_ch06_cache as C  # noqa: E402
import webapp_ch06_manifest as M  # noqa: E402
import webapp_ch06_paths as P  # noqa: E402
import webapp_ch06_server as S  # noqa: E402

OUT = HERE / "results" / "ch06.json"
PUBLIC = HERE / "pwa" / "public"
CH01_PUBLIC = HERE.parent / "ch01_everything_visible" / "editor" / "public"


def size(rel):
    return (PUBLIC / rel).stat().st_size


def e0_files():
    ch01_index = (CH01_PUBLIC / "index.html").read_text(encoding="utf-8").splitlines()
    ours = (PUBLIC / "index.html").read_text(encoding="utf-8").splitlines()
    added = [l for l in ours if l not in ch01_index]
    removed = [l for l in ch01_index if l not in ours]
    return {
        "index_html_bytes": size("index.html"),
        "ch01_index_html_bytes": (CH01_PUBLIC / "index.html").stat().st_size,
        "index_lines_added": added, "index_lines_removed": removed,
        "app_min_js_bytes": size("app.min.js"),
        "ch01_app_min_js_bytes": (CH01_PUBLIC / "app.min.js").stat().st_size,
        "manifest_bytes": size("manifest.json"), "sw_bytes": size("sw.js"),
        "icon_192_bytes": size("icons/icon-192.png"), "icon_512_bytes": size("icons/icon-512.png"),
        "manifest_keys": list(json.loads((PUBLIC / "manifest.json").read_text(encoding="utf-8")).keys()),
    }


def e1_manifests():
    out = {}
    cases = {"ours": PUBLIC / "manifest.json"}
    for p in sorted((HERE / "data" / "manifests").glob("*.json")):
        cases[p.stem] = p
    for name, path in cases.items():
        f = M.check_manifest(path.read_text(encoding="utf-8"), "http://127.0.0.1:8000/", PUBLIC)
        out[name] = {"chromium_installable": M.installable_in_chromium(f), "failed": M.failed(f),
                     "failed_messages": [x["message"] for x in f if not x["ok"]], "checks": len(f)}
    out["ours_on_plain_http"] = M.failed(M.check_manifest((PUBLIC / "manifest.json").read_text(encoding="utf-8"),
                                                        "http://card.example/", PUBLIC))
    return out


def e2_precache():
    cfg = C.load_sw_config()
    rows = [{"path": p, "bytes": C.file_bytes(p)} for p in cfg["precache"]]
    page = [{"path": p, "bytes": C.file_bytes(p)} for p in C.PAGE_GETS]
    return {"version": cfg["version"], "files": len(rows), "rows": rows, "total_bytes": sum(r["bytes"] for r in rows),
            "page_files": len(page), "page_bytes": sum(r["bytes"] for r in page), "page_rows": page,
            "extra_beyond_page": [r for r in rows if r["path"] not in C.PAGE_GETS],
            "routes": cfg["routes"]}


def e3_first_visit():
    r = C.run("mixed")["visits"][0]
    cfg = C.load_sw_config()
    return {"page_gets": len(C.PAGE_GETS), "post": 1, "precache": len(cfg["precache"]),
            "network_requests": r["network_requests"], "network_bytes": r["network_bytes"]}


def e4_strategies():
    out = {"scenario": ["온라인 첫 방문", "오프라인", "배포: " + ", ".join(C.DEPLOY_CHANGED), "온라인", "온라인", "오프라인"],
           "deploy_changed": C.DEPLOY_CHANGED, "runs": {}}
    for s in C.STRATEGIES:
        for bump in (True, False):
            r = C.run(s, bump)
            key = f"{s}|{'bump' if bump else 'no-bump'}"
            out["runs"][key] = {
                "total": r["total"],
                "per_visit": [{k: v[k] for k in ("online", "page_opened", "network_requests", "network_bytes",
                                                   "from_cache", "stale", "stale_paths", "failed")} for v in r["visits"]],
            }
    return out


def e5_paths():
    data = P.load_paths()
    table = [{k: p[k] for k in ("way", "name", "ui", "code", "store", "device", "can")} for p in data["paths"]]
    return {"conditions": data["conditions"], "table": table,
            "cases": {name: P.choose(needs, data) for name, needs in P.CASES.items()}}


def e6_crosscheck():
    cfg = C.load_sw_config()
    missing = [p for p in cfg["precache"] if not (PUBLIC / ("index.html" if p == "/" else p.lstrip("/"))).is_file()]
    mismatched = []
    with S.running() as base:
        for p in cfg["precache"] + ["/sw.js"]:
            body = urllib.request.urlopen(base + p).read()
            disk = (PUBLIC / ("index.html" if p == "/" else p.lstrip("/"))).read_bytes()
            if body != disk:
                mismatched.append(p)
        ctype = urllib.request.urlopen(base + "/manifest.json").headers["Content-Type"]
        sw_cache = urllib.request.urlopen(base + "/sw.js").headers["Cache-Control"]
    node = shutil.which("node")
    node_ok = None
    if node:
        node_ok = all(subprocess.run([node, "--check", str(PUBLIC / f)], capture_output=True).returncode == 0
                      for f in ("sw.js", "app.min.js"))
    return {"precache_missing": missing, "served_mismatch": mismatched, "manifest_content_type": ctype,
            "sw_cache_control": sw_cache, "node_check": node_ok,
            "sourcemap_line_in_app_min_js": "sourceMappingURL" in (PUBLIC / "app.min.js").read_text(encoding="utf-8")}


def main():
    res = {
        "meta": {"checked": "2026-10-10", "python": platform.python_version(),
                 "note": "숫자는 이 폴더의 파일과 모형에서 나온 값이에요. 브라우저 실측은 checks/ch06.csv에 적어요."},
        "e0_files": e0_files(),
        "e1_manifests": e1_manifests(),
        "e2_precache": e2_precache(),
        "e3_first_visit": e3_first_visit(),
        "e4_strategies": e4_strategies(),
        "e5_paths": e5_paths(),
        "e6_crosscheck": e6_crosscheck(),
    }
    res["meta"]["python"] = "3.12"  # 결과 파일이 버전 패치 번호로 바뀌지 않게
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("e0_files", "e2_precache", "e3_first_visit")}, ensure_ascii=False, indent=1)[:3000])
    for k, v in r["e4_strategies"]["runs"].items():
        print(k, v["total"])
    print(r["e1_manifests"])
    print(r["e6_crosscheck"])
