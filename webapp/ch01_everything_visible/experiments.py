"""1인 개발 앱 출시 실전 1장 실험. `uv run python webapp/ch01_everything_visible/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 처음 열 때 받는 파일 수와 바이트 합
E1 요청 목록: 처음 열기, 템플릿 바꾸기, AI 배경 만들기(종류, 시작한 곳, 크기, 압축 시 크기, 캐시 규칙)
E2 내려간 것과 서버에만 남은 것: editor/ 아래 모든 파일
E3 서버에만 둔 값이 받은 바이트 안에 있는가(가짜 키, 생성 지시문, 원본 템플릿의 작업 메모)
E4 압축(minify)과 이름 바꾸기: 크기, 주석 수, 그대로 남은 문자열
E5 지도 파일(source map): 내보낼 때와 안 내보낼 때
E6 브라우저 캐시 모형: 첫 방문 뒤 사용자 컴퓨터에 남는 것
E7 공개 폴더 밖 주소와 폴더 목록 요청
E8 대조: HTTP로 받은 크기 = 디스크 파일 크기, 압축 크기 재계산

표준 라이브러리만 쓴다. 서버는 127.0.0.1의 빈 포트에 잠깐 띄웠다 끈다. 결과는 매번 같다.
"""

import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import webapp_ch01_build as B  # noqa: E402
import webapp_ch01_server as S  # noqa: E402
import webapp_ch01_visible as V  # noqa: E402

OUT = HERE / "results" / "ch01.json"
SERVER_ONLY_MARKERS = {
    "가짜 AI 키": "DEMO_NOT_A_REAL_KEY_0000",
    "이미지 생성 지시문": "글자를 넣지 말 것",
    "원본 템플릿의 작업 메모": "작업 원본: 공지형 템플릿 v3",
    "하루 사용 한도 설정 이름": "daily_limit_per_user",
}


def e0_e1(log):
    rows = V.public_view(log)
    first = [r for r in rows if r["step"] == "open"]
    by_kind = {}
    for r in first:
        by_kind.setdefault(r["kind"], {"files": 0, "bytes": 0})
        by_kind[r["kind"]]["files"] += 1
        by_kind[r["kind"]]["bytes"] += r["bytes"]
    e0 = {"open_files": len(first), "open_bytes": sum(r["bytes"] for r in first),
          "open_gzip_bytes": sum(r["gzip_bytes"] for r in first), "by_kind": by_kind,
          "all_requests": len(rows), "all_bytes": sum(r["bytes"] for r in rows)}
    e1 = [{k: r[k] for k in ("step", "method", "path", "kind", "initiator", "status", "bytes", "gzip_bytes",
                             "cache_control")} for r in rows]
    return e0, e1


def e3_markers(log):
    blob = V.all_downloaded_bytes(log)
    return {name: blob.count(mark.encode()) for name, mark in SERVER_ONLY_MARKERS.items()}


def e4_minify():
    original, minified, smap = B.build(write=False)
    ob, mb = original.encode(), minified.encode()
    lits = [s for s in B.string_literals(original) if "templates/" in s or "/api/" in s]
    return {
        "original_bytes": len(ob), "min_bytes": len(mb), "min_ratio": round(len(mb) / len(ob), 4),
        "original_gzip": V.gzip_size(ob), "min_gzip": V.gzip_size(mb),
        "comments_original": len(B.comments(original)), "comments_min": len(B.comments(minified)),
        "comments_min_text": B.comments(minified),
        "renamed": len(B.RENAME),
        "path_strings_original": lits,
        "path_strings_kept_in_min": [s for s in lits if s in minified],
        "public_file_matches_build": (B.PUBLIC / B.MIN_NAME).read_text(encoding="utf-8") == minified
                                      and (B.PUBLIC / B.MAP_NAME).read_text(encoding="utf-8") == smap,
    }


def e5_source_maps():
    out = {}
    original = B.SRC.read_text(encoding="utf-8")
    for serve in (True, False):
        with V.running_editor(serve_maps=serve) as base:
            log = V.visit(base, actions=(), devtools_source_maps=True)
        m = [r for r in log if r["kind"] == "sourcemap"][0]
        rec = {"status": m["status"]}
        if m["status"] == 200:
            rec["bytes"] = m["bytes"]
            content = json.loads(m["_body"])["sourcesContent"][0]
            rec["original_recovered"] = content == original
            rec["comments_recovered"] = len(B.comments(content))
            rec["names_listed"] = len(json.loads(m["_body"])["names"])
        out["maps_served" if serve else "maps_hidden"] = rec
    return out


def e7_outside(base):
    paths = ["/server_only/ai_config.json", "/src/app.js", "/templates/", "/images/",
             "/api/generate-image"]
    return {p: V.request(base, p)[0] for p in paths}


def e8_crosscheck(log):
    bad = 0
    checked = 0
    for r in log:
        if r["method"] != "GET" or r["status"] != 200:
            continue
        rel = "index.html" if r["path"] == "/" else r["path"].lstrip("/")
        disk = (S.PUBLIC / rel).read_bytes()
        checked += 1
        if len(disk) != r["bytes"] or V.gzip_size(disk) != r["gzip_bytes"]:
            bad += 1
    return {"checked": checked, "mismatches": bad}


def main():
    with V.running_editor(serve_maps=True) as base:
        log = V.visit(base)
        log_dev = V.visit(base, devtools_source_maps=True)
        outside = e7_outside(base)
    e0, e1 = e0_e1(log)
    inv = V.inventory(log)
    inv_dev = {r["file"]: r["where"] for r in V.inventory(log_dev)}
    for r in inv:
        r["where_with_devtools_maps"] = inv_dev[r["file"]]
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "checked": "2026-10-10", "server": "http.server (ThreadingHTTPServer, 127.0.0.1)"},
        "e0_first_open": e0,
        "e1_requests": e1,
        "e2_inventory": inv,
        "e2_summary": {
            "downloaded": sum(1 for r in inv if r["where"] == "내려감"),
            "via_source_map_with_devtools": sum(1 for r in inv if r["where_with_devtools_maps"].startswith("지도")),
            "public_not_requested": sum(1 for r in inv if r["where"].startswith("공개")),
            "server_only": sum(1 for r in inv if r["where"] == "서버에만 있음"),
            "server_only_bytes": sum(r["bytes"] for r in inv if r["where"] == "서버에만 있음"),
        },
        "e3_server_only_markers_in_downloads": e3_markers(log),
        "e4_minify": e4_minify(),
        "e5_source_maps": e5_source_maps(),
        "e6_cache": V.cache_after_first_visit(log),
        "e7_outside_public": outside,
        "e8_crosscheck": e8_crosscheck(log),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
