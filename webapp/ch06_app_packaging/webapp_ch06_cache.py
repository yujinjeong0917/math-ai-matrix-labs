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

# none = 서비스 워커 없음(비교 기준), mixed = sw.js의 routes 표대로 경로마다 다른 전략.
STRATEGIES = ("none", "cache-first", "network-first", "stale-while-revalidate", "mixed")


def load_sw_config(text=None):
    text = SW.read_text(encoding="utf-8") if text is None else text
    # sw.js 안의 CONFIG 객체 리터럴만 정규식으로 잘라 JSON으로 읽는다(JS 전체를 해석하지 않는다. 그래서 config 블록은 JSON으로도 유효해야 한다).
    m = re.search(r"/\* config:start \*/\s*const CONFIG = (\{.*?\});\s*/\* config:end \*/", text, re.S)
    if not m:
        raise ValueError("sw.js 에 config 블록이 없어요")
    return json.loads(m.group(1))


def file_bytes(path):
    # "/"는 디스크에서 index.html. 크기는 압축 전 바이트.
    rel = "index.html" if path == "/" else path.lstrip("/")
    return (PUBLIC / rel).stat().st_size


def route_strategy(config, path, navigate):
    # routes는 위에서부터 차례로 보고 처음 맞는 규칙을 쓴다(순서가 의미). navigate는 문서 이동 요청에만 맞는다.
    for r in config["routes"]:
        if r["match"] == "navigate":
            if navigate:
                return r["strategy"]
        elif r["match"] == "*" or path.startswith(r["match"]):
            return r["strategy"]
    # 어떤 규칙에도 안 맞을 때의 기본값(안전하게 캐시를 쓰지 않음).
    return "network-only"


class World:
    """서버(파일 버전)와 한 사용자의 브라우저(HTTP 캐시, 서비스 워커, Cache Storage)."""

    def __init__(self, strategy, config=None):
        assert strategy in STRATEGIES
        self.strategy = strategy
        self.config = config or load_sw_config()
        # 서버의 파일별 버전 번호(처음 1). 배포하면 올라간다. 캐시가 가진 번호와 비교해 옛 버전인지(stale) 판정한다.
        self.server = {p: 1 for p in set(PAGE_GETS) | set(self.config["precache"])}
        self.server_sw = 1                      # 서버에 올라간 sw.js 의 버전
        # none 전략에서만 쓰는 브라우저 HTTP 캐시(1장의 하루 max-age 규칙).
        self.http_cache = {}                    # path -> 버전 (none 전략에서만 씀)
        self.sw_active = None                   # 활성 서비스 워커의 버전
        # 새 서비스 워커가 설치를 마치고 기다리는 상태: (버전, 새 캐시). 다음 방문 시작에 활성화된다.
        self.sw_waiting = None                  # 설치를 마치고 기다리는 버전과 그 캐시
        self.cache = {}                         # Cache Storage: path -> 버전

    # ---------- 배포 ----------
    # bump_sw=False는 파일은 바꿨는데 sw.js(버전)를 안 바꾼 실수를 재현한다. 브라우저가 sw.js가 바뀐 것을 못 봐 새 캐시를 만들지 않는다.
    def deploy(self, changed, bump_sw=True):
        for p in changed:
            self.server[p] += 1
        if bump_sw:
            self.server_sw += 1

    # ---------- 방문 ----------
    def visit(self, online):
        # 방문 한 번의 집계. stale = 서버의 최신 버전과 다른 것을 받은 수, failed = 못 받은 요청 수.
        rec = {"online": online, "network_requests": 0, "network_bytes": 0, "from_cache": 0,
               "stale": 0, "failed": 0, "page_opened": False, "stale_paths": [], "background_requests": 0}

        def net(path):
            rec["network_requests"] += 1
            rec["network_bytes"] += file_bytes(path)
            return self.server[path]

        def served(path, version, from_cache):
            if from_cache:
                rec["from_cache"] += 1
            # 받은 버전이 서버 최신과 다르면 옛 버전을 준 것.
            if version != self.server[path]:
                rec["stale"] += 1
                rec["stale_paths"].append(path)

        if self.sw_waiting is not None:                         # 기다리던 새 서비스 워커가 이번 방문부터 맡는다
            self.sw_active, self.cache = self.sw_waiting
            self.sw_waiting = None

        # 첫 방문에는 sw_active가 없어 이 방문은 서비스 워커가 맡지 않는다(그래서 모든 요청이 네트워크).
        controlled = self.strategy != "none" and self.sw_active is not None
        for path in PAGE_GETS:
            navigate = path == "/"
            if not controlled:
                ok = self._http_get(path, online, net, served)
            else:
                ok = self._sw_get(path, navigate, online, net, served, rec)
            if not ok:
                rec["failed"] += 1
                # 문서(/)를 못 받으면 이후 요청을 보지 않고 끝낸다(페이지가 열리지 않음).
                if navigate:                                    # 문서를 못 받으면 페이지가 안 열린다
                    return rec
            elif navigate:
                rec["page_opened"] = True

        # POST는 Cache API에 넣을 수 없어 어느 전략이든 온라인일 때만 된다(모형의 가정, 위 docstring 참고).
        if online:                                              # AI 배경 버튼: POST 는 캐시에 못 넣는다
            rec["network_requests"] += 1
        else:
            rec["failed"] += 1

        if self.strategy != "none" and online:
            if self.sw_active is None:                          # 첫 방문: 등록하고 설치(precache)
                # 첫 방문: 설치 단계에서 precache 12개를 네트워크로 받는다(페이지 8개와 겹쳐도 다시 받는 것으로 센다).
                self.cache = {p: net(p) for p in self.config["precache"]}
                self.sw_active = self.server_sw
            else:                                               # 업데이트 확인: sw.js 를 받아 비교
                rec["network_requests"] += 1
                # 온라인 방문마다 sw.js를 한 번 받아 버전 비교(sw.js는 no-cache). 업데이트 확인의 비용.
                rec["network_bytes"] += SW.stat().st_size
                # sw.js가 바뀌었을 때만 새 서비스 워커가 설치(precache 새로 받기)를 시작한다.
                if self.server_sw != self.sw_active:
                    self.sw_waiting = (self.server_sw, {p: net(p) for p in self.config["precache"]})
        return rec

    def _http_get(self, path, online, net, served):
        # none 전략: HTML은 no-cache라 매번 서버에 확인하고 받는다(여기서는 네트워크로 센다).
        if path == "/":                                         # no-cache: 쓰기 전에 서버에 확인
            if not online:
                return False
            served(path, net(path), False)
            return True
        # 나머지는 max-age(하루)라 묻지 않고 저장본. 배포로 바뀌어도 하루 동안은 옛것을 준다(stale).
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
        # mixed면 요청마다 routes 표로 전략을 고른다.
        strategy = self.strategy if self.strategy != "mixed" else route_strategy(self.config, path, navigate)
        hit = self.cache.get(path)
        # cache-first: 캐시에 있으면 그것을 주고 네트워크를 보지 않는다. 가장 빠르지만 버전을 바꾸지 않으면 계속 옛것.
        if strategy == "cache-first":
            if hit is not None:
                served(path, hit, True)
                return True
            if not online:
                return False
            self.cache[path] = net(path)
            served(path, self.cache[path], False)
            return True
        # network-first: 온라인이면 항상 새로 받고(캐시도 갱신), 오프라인일 때만 캐시.
        if strategy == "network-first":
            if online:
                self.cache[path] = net(path)
                served(path, self.cache[path], False)
                return True
            if hit is not None:
                served(path, hit, True)
                return True
            return False
        # SWR: 캐시본을 먼저 주고 뒤에서 새로 받아 캐시만 고친다. 이번 방문은 옛것, 다음 방문부터 새것. 요청 수는 network-first와 같다.
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
# 배포에서 바뀌는 파일 3개(문서, 템플릿 JSON, 앱 코드). 전략별 stale 수의 기대값이 여기서 나온다.
DEPLOY_CHANGED = ["/", "/templates/notice.json", "/app.min.js"]
# 방문 5번 + 배포 1번: 첫 온라인, 오프라인, (배포), 온라인, 온라인, 오프라인.
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
    # 방문별 집계를 합산. page_opened는 불리언이라 sum이 열린 횟수가 된다.
    total = {k: sum(v[k] for v in visits) for k in keys}
    total["pages_opened"] = sum(v["page_opened"] for v in visits)
    return {"strategy": strategy, "bump_sw": bump_sw, "visits": visits, "total": total}


if __name__ == "__main__":
    for s in STRATEGIES:
        for bump in (True, False):
            r = run(s, bump)
            print(s, "bump" if bump else "no-bump", [(v["page_opened"], v["stale"], v["failed"]) for v in r["visits"]], r["total"])
