"""1인 개발 앱 출시 실전 6장: 오프라인 캐시 전략 모형.

브라우저와 서비스 워커를 흉내 낸 작은 상태 기계다. 진짜 브라우저를 띄우지 않고, 방문 순서와 네트워크 상태,
배포(파일이 바뀜)를 정해 두고 전략마다 "네트워크 요청 수, 캐시에서 준 수, 옛 버전을 준 수, 실패 수"를 센다.

전략 규칙은 pwa/public/sw.js 의 /* config:start */ ~ /* config:end */ 블록을 그대로 읽어 쓴다.
그래서 sw.js 를 고치면 모형도 함께 바뀐다(둘이 따로 놀지 않게).

모형의 가정
- 첫 방문에는 서비스 워커가 아직 없어서 모든 요청이 네트워크로 간다. 페이지가 뜬 뒤 등록·설치하며
  precache 목록을 네트워크에서 받아 저장한다(실제로는 HTTP 캐시에서 꺼낼 수도 있다). 다음 방문부터 서비스 워커가 맡는다.
- 온라인 방문마다 브라우저가 sw.js 를 한 번 받아 바뀌었는지 본다. 바뀌었으면 새 서비스 워커가 precache 를 새로 받고
  기다리다가, 다음 방문 시작에 활성화되며 옛 캐시를 지운다(방문 사이에 탭을 모두 닫는다고 둔다).
- "none"(서비스 워커 없음)은 1장의 캐시 규칙만 쓴다: HTML은 no-cache(쓰기 전에 서버 확인), 나머지는 하루 동안 저장본 사용.
  모든 방문이 하루 안에 일어난다고 둔다.
- 서비스 워커의 네트워크 요청(설치 때 미리 받기, fetch)은 HTTP 캐시를 건너뛰고 서버에서 바로 받는다고 둔다.
  sw.js 가 { cache: "reload" } 로 그렇게 요청하기 때문이다. 이 옵션이 없으면 기본값(default)이라 하루짜리 HTTP 캐시에서
  옛 파일을 꺼낼 수 있고, 그러면 network-first 도 옛 버전을 줄 수 있다(MDN Request.cache).
- POST(/api/generate-image)는 Cache API에 넣을 수 없어서(Service Workers 명세: GET이 아니면 TypeError) 어느 전략이든
  온라인일 때만 된다.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
PUBLIC = HERE / "pwa" / "public"
SW = PUBLIC / "sw.js"

# 한 번 방문할 때 페이지가 요청하는 것(1장 편집기와 같은 8개) + AI 배경 버튼 한 번
PAGE_GETS = ["/", "/images/logo.svg", "/style.css", "/fonts/card-sans.woff2", "/images/bg-dots.svg",
             "/app.min.js", "/templates/index.json", "/templates/notice.json"]
ACTION_POST = "/api/generate-image"

STRATEGIES = ("none", "cache-first", "network-first", "stale-while-revalidate", "mixed")


def load_sw_config(text=None):
    text = SW.read_text(encoding="utf-8") if text is None else text
    m = re.search(r"/\* config:start \*/\s*const CONFIG = (\{.*?\});\s*/\* config:end \*/", text, re.S)
    if not m:
        raise ValueError("sw.js 에 config 블록이 없어요")
    return json.loads(m.group(1))


def file_bytes(path):
    rel = "index.html" if path == "/" else path.lstrip("/")
    return (PUBLIC / rel).stat().st_size


def route_strategy(config, path, navigate):
    for r in config["routes"]:
        if r["match"] == "navigate":
            if navigate:
                return r["strategy"]
        elif r["match"] == "*" or path.startswith(r["match"]):
            return r["strategy"]
    return "network-only"


class World:
    """서버(파일 버전)와 한 사용자의 브라우저(HTTP 캐시, 서비스 워커, Cache Storage)."""

    def __init__(self, strategy, config=None):
        assert strategy in STRATEGIES
        self.strategy = strategy
        self.config = config or load_sw_config()
        self.server = {p: 1 for p in set(PAGE_GETS) | set(self.config["precache"])}
        self.server_sw = 1                      # 서버에 올라간 sw.js 의 버전
        self.http_cache = {}                    # path -> 버전 (none 전략에서만 씀)
        self.sw_active = None                   # 활성 서비스 워커의 버전
        self.sw_waiting = None                  # 설치를 마치고 기다리는 버전과 그 캐시
        self.cache = {}                         # Cache Storage: path -> 버전

    # ---------- 배포 ----------
    def deploy(self, changed, bump_sw=True):
        for p in changed:
            self.server[p] += 1
        if bump_sw:
            self.server_sw += 1

    # ---------- 방문 ----------
    def visit(self, online):
        rec = {"online": online, "network_requests": 0, "network_bytes": 0, "from_cache": 0,
               "stale": 0, "failed": 0, "page_opened": False, "stale_paths": [], "background_requests": 0}

        def net(path):
            rec["network_requests"] += 1
            rec["network_bytes"] += file_bytes(path)
            return self.server[path]

        def served(path, version, from_cache):
            if from_cache:
                rec["from_cache"] += 1
            if version != self.server[path]:
                rec["stale"] += 1
                rec["stale_paths"].append(path)

        if self.sw_waiting is not None:                         # 기다리던 새 서비스 워커가 이번 방문부터 맡는다
            self.sw_active, self.cache = self.sw_waiting
            self.sw_waiting = None

        controlled = self.strategy != "none" and self.sw_active is not None
        for path in PAGE_GETS:
            navigate = path == "/"
            if not controlled:
                ok = self._http_get(path, online, net, served)
            else:
                ok = self._sw_get(path, navigate, online, net, served, rec)
            if not ok:
                rec["failed"] += 1
                if navigate:                                    # 문서를 못 받으면 페이지가 안 열린다
                    return rec
            elif navigate:
                rec["page_opened"] = True

        if online:                                              # AI 배경 버튼: POST 는 캐시에 못 넣는다
            rec["network_requests"] += 1
        else:
            rec["failed"] += 1

        if self.strategy != "none" and online:
            if self.sw_active is None:                          # 첫 방문: 등록하고 설치(precache)
                self.cache = {p: net(p) for p in self.config["precache"]}
                self.sw_active = self.server_sw
            else:                                               # 업데이트 확인: sw.js 를 받아 비교
                rec["network_requests"] += 1
                rec["network_bytes"] += SW.stat().st_size
                if self.server_sw != self.sw_active:
                    self.sw_waiting = (self.server_sw, {p: net(p) for p in self.config["precache"]})
        return rec

    def _http_get(self, path, online, net, served):
        if path == "/":                                         # no-cache: 쓰기 전에 서버에 확인
            if not online:
                return False
            served(path, net(path), False)
            return True
        if path in self.http_cache:                             # max-age 하루: 묻지 않고 저장본
            served(path, self.http_cache[path], True)
            return True
        if not online:
            return False
        v = net(path)
        self.http_cache[path] = v
        served(path, v, False)
        return True

    def _sw_get(self, path, navigate, online, net, served, rec):
        strategy = self.strategy if self.strategy != "mixed" else route_strategy(self.config, path, navigate)
        hit = self.cache.get(path)
        if strategy == "cache-first":
            if hit is not None:
                served(path, hit, True)
                return True
            if not online:
                return False
            self.cache[path] = net(path)
            served(path, self.cache[path], False)
            return True
        if strategy == "network-first":
            if online:
                self.cache[path] = net(path)
                served(path, self.cache[path], False)
                return True
            if hit is not None:
                served(path, hit, True)
                return True
            return False
        if strategy == "stale-while-revalidate":
            if hit is not None:
                served(path, hit, True)
                if online:                                      # 뒤에서 새로 받아 캐시만 고친다
                    self.cache[path] = net(path)
                    rec["background_requests"] += 1
                return True
            if not online:
                return False
            self.cache[path] = net(path)
            served(path, self.cache[path], False)
            return True
        if online:                                              # network-only
            served(path, net(path), False)
            return True
        return False


# 기본 시나리오: 첫 방문 → 지하철(오프라인) → 배포 → 두 번 온라인 → 다시 오프라인
DEPLOY_CHANGED = ["/", "/templates/notice.json", "/app.min.js"]
SCENARIO = [("visit", True), ("visit", False), ("deploy", None), ("visit", True), ("visit", True), ("visit", False)]


def run(strategy, bump_sw=True, scenario=SCENARIO, changed=DEPLOY_CHANGED):
    w = World(strategy)
    visits = []
    for step, arg in scenario:
        if step == "deploy":
            w.deploy(changed, bump_sw=bump_sw)
        else:
            visits.append(w.visit(arg))
    keys = ("network_requests", "network_bytes", "from_cache", "stale", "failed", "background_requests")
    total = {k: sum(v[k] for v in visits) for k in keys}
    total["pages_opened"] = sum(v["page_opened"] for v in visits)
    return {"strategy": strategy, "bump_sw": bump_sw, "visits": visits, "total": total}


if __name__ == "__main__":
    for s in STRATEGIES:
        for bump in (True, False):
            r = run(s, bump)
            print(s, "bump" if bump else "no-bump", [(v["page_opened"], v["stale"], v["failed"]) for v in r["visits"]], r["total"])
