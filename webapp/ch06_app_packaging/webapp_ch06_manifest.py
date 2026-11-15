"""1인 개발 앱 출시 실전 6장: manifest.json 검사기.

명세(W3C Web Application Manifest)에서는 모든 항목이 선택이다. "필수"는 브라우저가 정한다.
이 검사기는 세 묶음으로 나눠 알린다.

- chromium : 크롬·엣지·삼성 인터넷이 설치를 제안하는 조건(MDN "Making PWAs installable", web.dev "install-criteria",
             확인일 2026-10-10). 하나라도 어기면 그 브라우저는 설치를 제안하지 않는다.
- file     : manifest에 적은 아이콘 파일이 실제로 있고, 적은 크기와 실제 그림 크기가 같은가(이 실습이 덧붙인 대조).
- spec     : 명세의 처리 규칙 때문에 적은 값이 무시되는 경우(start_url이 scope 밖이면 scope가 무시됨).

Safari(iOS·iPadOS 26)는 manifest가 없어도 홈 화면에 추가한 사이트를 웹앱으로 연다(WebKit 블로그, 2025-09-15).
그래서 이 검사기는 "Safari에서 설치되는가"를 판정하지 않는다.
"""

import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

import webapp_ch06_icons as I

HERE = Path(__file__).parent
PUBLIC = HERE / "pwa" / "public"

# 크롬 계열이 설치를 제안하는 display 값 목록(출처: 위 docstring의 MDN, web.dev).
INSTALL_DISPLAY = ("fullscreen", "standalone", "minimal-ui", "window-controls-overlay")
# 192x192와 512x512 두 크기가 모두 있어야 설치 조건 충족.
REQUIRED_ICON_SIZES = ("192x192", "512x512")


def _declared_sizes(icons):
    out = set()
    # manifest가 망가져 icons가 목록이 아니어도 검사기가 죽지 않게 방어.
    for icon in icons if isinstance(icons, list) else []:
        if isinstance(icon, dict):
            out.update(str(icon.get("sizes", "")).split())
    return out


def secure_context(page_url):
    """https 이거나 내 컴퓨터(localhost, 127.0.0.1)면 True. 포트는 상관없다."""
    u = urlparse(page_url)
    # localhost는 예외로 https 없이도 서비스 워커가 등록된다. 같은 와이파이의 192.168.x.x는 예외가 아니다(test가 확인).
    return u.scheme == "https" or (u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1"))


def check_manifest(text, page_url="https://card.example/", public_dir=None):
    """manifest 글(text)을 검사해 [{group, id, ok, message}] 목록을 돌려준다."""
    findings = []

    def add(group, fid, ok, message):
        findings.append({"group": group, "id": fid, "ok": bool(ok), "message": message})

    try:
        # JSON 문법 오류는 가장 먼저 보고하고 끝낸다(뒤의 검사가 의미 없음). 끝 쉼표 같은 실수를 잡는다.
        m = json.loads(text)
    except json.JSONDecodeError as err:
        add("chromium", "json", False, f"JSON으로 읽을 수 없어요({err.msg}, {err.lineno}번째 줄)")
        return findings
    if not isinstance(m, dict):
        add("chromium", "json", False, "manifest는 { } 로 감싼 객체여야 해요")
        return findings

    add("chromium", "https", secure_context(page_url), "https 이거나 localhost·127.0.0.1 이어야 해요")
    # name과 short_name 중 하나라도 비어 있지 않은 문자열이면 통과.
    name_ok = any(isinstance(m.get(k), str) and m.get(k).strip() for k in ("name", "short_name"))
    add("chromium", "name", name_ok, "name 이나 short_name 중 하나는 있어야 해요")
    sizes = _declared_sizes(m.get("icons"))
    missing = [s for s in REQUIRED_ICON_SIZES if s not in sizes]
    add("chromium", "icons", not missing,
        "아이콘 192x192와 512x512가 모두 있어야 해요" + (f"(빠진 크기: {', '.join(missing)})" if missing else ""))
    add("chromium", "start_url", isinstance(m.get("start_url"), str) and m.get("start_url") != "",
        "start_url 이 있어야 해요")
    # display_override도 함께 보므로, display가 browser여도 override에 standalone이 있으면 통과(test_display_override_counts).
    displays = [m.get("display")]
    if isinstance(m.get("display_override"), list):
        displays += m["display_override"]
    add("chromium", "display", any(d in INSTALL_DISPLAY for d in displays),
        "display(또는 display_override)가 fullscreen, standalone, minimal-ui, window-controls-overlay 중 하나여야 해요")
    add("chromium", "prefer_related_applications", m.get("prefer_related_applications") is not True,
        "prefer_related_applications 는 없거나 false 여야 해요")

    if public_dir is not None:
        for k, icon in enumerate(m.get("icons", []) if isinstance(m.get("icons"), list) else []):
            src = str(icon.get("src", ""))
            # 아이콘 src를 사이트 루트 기준 경로로 바꿔 디스크 파일에 대응시킨다(쿼리스트링 제거).
            path = Path(public_dir) / urlparse(urljoin("/", src)).path.lstrip("/")
            if not path.is_file():
                add("file", f"icon{k}:{src}", False, f"{src} 파일이 없어요")
                continue
            # 적은 sizes와 실제 PNG 크기가 같은지 본다. 크기 판정은 크롬 조건이 아니라 이 실습이 덧붙인 확인(file 그룹).
            real = I.png_size(path.read_bytes()) if src.endswith(".png") else None
            declared = str(icon.get("sizes", ""))
            ok = real is not None and f"{real[0]}x{real[1]}" in declared.split()
            got = f"{real[0]}x{real[1]}" if real else "PNG가 아님"
            add("file", f"icon{k}:{src}", ok, f"{src}: 적은 크기 {declared}, 실제 {got}")

    if isinstance(m.get("scope"), str) and isinstance(m.get("start_url"), str):
        scope = urljoin(page_url, m["scope"])
        start = urljoin(page_url, m["start_url"])
        # scope 안인지 단순 비교: 같은 출처이고 start_url 경로가 scope 경로로 시작하는지(정확한 명세 처리보다 단순화).
        within = urlparse(start).netloc == urlparse(scope).netloc and urlparse(start).path.startswith(urlparse(scope).path)
        add("spec", "scope", within, "start_url 이 scope 안에 있어야 scope 가 쓰여요(밖이면 scope 를 무시해요)")
    return findings


def installable_in_chromium(findings):
    # 크롬 조건 그룹만으로 판정. file, spec 그룹의 실패는 설치 가능 여부와 별개.
    return all(f["ok"] for f in findings if f["group"] == "chromium")


def failed(findings):
    return [f["id"] for f in findings if not f["ok"]]


if __name__ == "__main__":
    res = check_manifest((PUBLIC / "manifest.json").read_text(encoding="utf-8"), "http://127.0.0.1:8000/", PUBLIC)
    for f in res:
        print("통과" if f["ok"] else "실패", f["group"], f["message"])
