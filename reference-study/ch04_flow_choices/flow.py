"""4장: 화면마다 사용자가 마주치는 링크를 세고, 목표까지 몇 걸음인지 센다.

입력은 links_snapshot_2026-11-18.json 하나다. 이 파일은 사이트의 세 화면 파일(홈, 학습 트랙 목록 화면, 트랙 장 화면 06번)에서
<main> 안의 <a href> 를 파서로 뽑아 저장한 스냅샷이고, 뽑은 일회성 스크립트는 이 폴더에 두지 않았다.
이 코드는 사이트 파일을 읽지 않고 스냅샷만 읽는다.

링크 수는 "선택지가 있을 수 있는 곳"의 상한이지 사람이 실제로 비교하는 선택의 수가 아니다.
사용자가 어디서 망설이는지는 이 실습이 재지 않는다.
"""
import json
import os
import re
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, "links_snapshot_2026-11-18.json")
ASSET_EXT = (".webp", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf")
CHAPTER_PATTERN = re.compile(r"^/tracks/[^/]+/\d\d-[^/]+\.html$")

# 화면이 요구하는 것에 대한 글쓴이의 한 문장 요약(추정). 형식: ~할지 ~할지(~할지)
DECISIONS = {
    "home": "사용자는 매트릭스로 갈지 학습 트랙으로 갈지 정한다",
    "tracks": "사용자는 어느 학습 트랙의 어느 장을 읽을지 정한다",
    "chapter": "사용자는 끝까지 읽을지 실습으로 갈지 다음 장으로 갈지 정한다",
}
LABELS = {"home": "홈", "tracks": "학습 트랙 목록 화면", "chapter": "트랙 장 화면(06번)"}


def load(path=SNAPSHOT):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["screens"]


def is_asset(resolved):
    return resolved.split("#")[0].lower().endswith(ASSET_EXT)


def page_of(resolved):
    return resolved.split("#")[0]


def counts(screen):
    """링크 수를 구분해 센다: 전체, 같은 화면 # 링크, 외부, 사이트 안(이 중 이미지 파일 링크)."""
    links = screen["links"]
    internal = [l for l in links if l["kind"] == "internal"]
    return {
        "total": len(links),
        "same_page": sum(1 for l in links if l["kind"] == "same_page"),
        "external": sum(1 for l in links if l["kind"] == "external"),
        "internal": len(internal),
        "internal_asset": sum(1 for l in internal if is_asset(l["resolved"])),
    }


def unique_addresses(screen):
    """사이트 안 링크의 고유 주소(# 뒤 조각 포함)."""
    return {l["resolved"] for l in screen["links"] if l["kind"] == "internal"}


def choices(screen):
    """사용자가 고르는 길: 사이트 안 링크에서 이미지 파일 링크를 뺀 것."""
    return [l for l in screen["links"] if l["kind"] == "internal" and not is_asset(l["resolved"])]


def graph(screens):
    """화면 파일 사이의 이동 관계: 화면 주소 -> 그 화면의 사이트 안 링크가 가는 화면 주소들(자기 자신, 이미지 파일 제외)."""
    g = {}
    for s in screens.values():
        targets = {page_of(l["resolved"]) for l in s["links"]
                   if l["kind"] == "internal" and not is_asset(l["resolved"])}
        targets.discard(s["page"])
        g[s["page"]] = targets
    return g


def shortest_path(g, start, goal):
    """너비 우선 탐색. 스냅샷에 담은 화면에서 나가는 선만 따라간다. 도달하지 못하면 None."""
    prev = {start: None}
    q = deque([start])
    while q:
        node = q.popleft()
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = prev[node]
            return path[::-1]
        for nxt in sorted(g.get(node, ())):
            if nxt not in prev:
                prev[nxt] = node
                q.append(nxt)
    return None


def target_breakdown(screen):
    """사이트 안 링크가 어디로 가는지 묶어 센다: 학습 트랙 목록 화면(/tracks/ 조각 포함), 개념 카드(/concepts/), 매트릭스(/matrix/)."""
    out = {"tracks": 0, "concepts": 0, "matrix": 0}
    for l in screen["links"]:
        if l["kind"] != "internal":
            continue
        r = l["resolved"]
        if page_of(r) == "/tracks/":
            out["tracks"] += 1
        elif r.startswith("/concepts/"):
            out["concepts"] += 1
        elif page_of(r) == "/matrix/":
            out["matrix"] += 1
    out["other"] = counts(screen)["internal"] - out["tracks"] - out["concepts"] - out["matrix"]
    return out


def chapter_links_by_track(screen):
    """장 화면(/tracks/<트랙>/NN-....html)로 가는 링크를 트랙별로 센다."""
    out = {}
    for l in screen["links"]:
        pg = page_of(l["resolved"])
        if l["kind"] == "internal" and CHAPTER_PATTERN.match(pg):
            out.setdefault(pg.split("/")[2], []).append(pg)
    return out


def direct_links_to_chapters(screen):
    """이 화면의 사이트 안 링크 중 트랙 장 화면(/tracks/<트랙>/NN-....html)로 가는 것."""
    return sorted({page_of(l["resolved"]) for l in screen["links"]
                   if l["kind"] == "internal" and CHAPTER_PATTERN.match(page_of(l["resolved"]))})


if __name__ == "__main__":
    sc = load()
    for k, v in sc.items():
        c = counts(v)
        print(f"{LABELS[k]}: 전체 {c['total']}개 = 같은 화면 # {c['same_page']} + 외부 {c['external']} + 사이트 안 {c['internal']}"
              f"(이미지 파일 {c['internal_asset']}), 고유 주소 {len(unique_addresses(v))}개")
    for k in sc:
        print(f"{LABELS[k]} | {DECISIONS[k]}")
    b = target_breakdown(sc["home"])
    print(f"홈의 사이트 안 링크 중 학습 트랙 목록 화면행 {b['tracks']}개(# 조각 링크 {b['tracks'] - 1}개 포함), 개념 카드행 {b['concepts']}개, 매트릭스행 {b['matrix']}개, 그 밖 {b['other']}개")
    oth = [page_of(l["resolved"]) for l in sc["home"]["links"] if l["kind"] == "internal"
           and page_of(l["resolved"]) not in ("/tracks/", "/matrix/") and not l["resolved"].startswith("/concepts/")]
    print(f"  그 밖 {len(oth)}개의 도착 화면 {len(set(oth))}곳: 두 번씩 나오는 화면 {sum(1 for x in set(oth) if oth.count(x) == 2)}곳, 한 번 나오는 화면 {sum(1 for x in set(oth) if oth.count(x) == 1)}곳")
    from collections import Counter
    rep = Counter(l["resolved"] for l in sc["home"]["links"] if l["kind"] == "internal")
    more = [(k, v) for k, v in rep.items() if v > 1]
    print(f"  홈에서 두 번 이상 나오는 주소 {len(more)}곳: 세 번 {sum(1 for _, v in more if v == 3)}곳, 두 번 {sum(1 for _, v in more if v == 2)}곳 -> 중복 {sum(v - 1 for _, v in more)}개")
    g = graph(sc)
    goal = sc["chapter"]["page"]
    cb = chapter_links_by_track(sc["tracks"])
    print(f"학습 트랙 목록 화면의 장 링크 {sum(len(v) for v in cb.values())}개는 학습 트랙 {len(cb)}곳에 나뉘어 있고, 한 트랙에 최대 {max(len(v) for v in cb.values())}개")
    print("홈의 장 화면 직접 링크:", direct_links_to_chapters(sc["home"]) or "없음")
    print("홈 파일 전체(전역 메뉴와 푸터 포함)에서 찾은 장 화면 주소:", sc["home"]["file_wide_chapter_hrefs"] or "없음")
    print("학습 트랙 목록 화면의 06번 장 링크:", goal in g[sc["tracks"]["page"]])
    path = shortest_path(g, sc["home"]["page"], goal)
    print(f"홈에서 장 화면까지 최소 {len(path) - 1}걸음: {' -> '.join(path)}")
    front = [k for k in sc if sc[k]["page"] in path[:-1]]
    s_links = sum(counts(sc[k])["internal"] for k in front)
    s_unique = sum(len(unique_addresses(sc[k])) for k in front)
    union = set().union(*(unique_addresses(sc[k]) for k in front))
    print(f"경로 앞 두 화면의 사이트 안 링크 합(상한): {s_links}")
    print(f"경로 앞 두 화면의 고유 주소 합: {s_unique}, 두 화면을 합친 고유 주소: {len(union)}, 겹치는 주소: {s_unique - len(union)}")
    for k in sc:
        c = counts(sc[k])["internal"]
        u = len(unique_addresses(sc[k]))
        print(f"{LABELS[k]}: 사이트 안 링크 {c}개 - 고유 주소 {u}개 = 중복 {c - u}개({(c - u) / c * 100:.1f}%)")
    h, t = counts(sc["home"])["internal"], counts(sc["tracks"])["internal"]
    print(f"학습 트랙 목록 / 홈 사이트 안 링크 비: {t / h:.2f}배")
    ch = sc["chapter"]
    n_ch = len(choices(ch))
    u_ch = len({l["resolved"] for l in choices(ch)})
    print(f"장 화면 사이트 안 {counts(ch)['internal']}개 = 고르는 길 {n_ch}개 + 이미지 파일 {counts(ch)['internal_asset']}개, 고르는 길의 고유 주소 {u_ch}개(중복 {n_ch - u_ch}개, {(n_ch - u_ch) / n_ch * 100:.1f}%)")
    print(f"트랙 목록 화면 / 장 화면 고르는 길 비: {t / n_ch:.2f}배")
