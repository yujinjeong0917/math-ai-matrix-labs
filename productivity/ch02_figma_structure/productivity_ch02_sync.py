"""생산성 도구 2장: 두 사람이 같은 시안을 고칠 때, 합치는 방식에 따라 무엇이 사라지나.

문서는 (레이어, 속성) 칸의 모음이다. 두 사람 A, B가 같은 원본에서 출발해 각자 k번 고친 뒤 합친다.
합치는 방식 넷을 같은 편집 기록에 적용한다.

- file   : 파일 통째로 마지막 저장이 이긴다. Figma가 멀티플레이어 전에 썼던 저장 방식
           ("문서를 내려받아 브라우저에서 고치고, 문서 전체를 주기적으로 다시 올린다")과
           Dropbox 같은 동기화 서비스로 파일을 주고받는 방식의 단순 모형.
           출처: Evan Wallace, "Multiplayer Editing in Figma", Figma 블로그, 2016-09-28.
- object : 레이어(객체) 단위로 나중 값이 이긴다. 비교용으로 만든 중간 단계 모형이다. [해석]
- prop   : (레이어, 속성) 단위로 서버에 마지막으로 도착한 값이 이긴다. Figma가 2016-09-28 동시 편집을
           공개할 때 이미 이 단위였다("... unless they affect the same property on the same object,
           in which case we pick the latest change", Wallace 2016). 자세한 설명은
           Evan Wallace, "How Figma's multiplayer technology works", Figma 블로그, 2019-10-16.
           "two clients changing unrelated properties on the same object won't conflict ..."
- baton  : 한 번에 한 사람만 고친다(잠금). Wallace(2016)가 검토했다고 쓴 'baton-passing' 방식의 모형.
           잃는 것은 없지만 B는 A가 끝날 때까지 기다린다.

센 값
- lost   : 그 사람이 고친 칸인데, 최종 문서에 그 값이 없고 다른 사람이 그 칸을 고친 적도 없는 경우.
           (누구도 덮어쓸 생각이 없었는데 사라진 변경)
- silent : 두 사람이 같은 칸을 고쳐서 한쪽 값이 말없이 덮인 경우. 알림이 없으면 둘 다 모른다.

실제 Figma 서버는 이 모형보다 훨씬 많은 경우(삭제, 부모 바꾸기의 순환, 확인 전 값의 깜박임 등)를
다룬다. 여기서는 '합치는 단위'의 차이만 떼어 본다.
"""

import random


def make_edits(n_layers, n_props, k, seed):
    """A와 B가 각각 k번 고친다. 칸은 무작위(중복 허용), 시각은 [0, 1) 무작위."""
    rng = random.Random(seed)
    edits = []
    for who in ("A", "B"):
        for i in range(k):
            edits.append({"t": rng.random(), "who": who, "cell": (rng.randrange(n_layers), rng.randrange(n_props)), "v": f"{who}{i}"})
    edits.sort(key=lambda e: e["t"])
    return edits


def _local(edits, who):
    """한 사람의 사본에 그 사람의 편집만 반영한 결과 {cell: value}."""
    out = {}
    for e in edits:
        if e["who"] == who:
            out[e["cell"]] = e["v"]
    return out


def merge(edits, mode, save_order=("A", "B")):
    """합친 결과 {cell: value}. 원본 값은 None으로 친다. save_order의 뒤쪽이 나중에 저장한다."""
    first, last = save_order
    a, b = _local(edits, first), _local(edits, last)
    if mode == "file":
        return dict(b)  # 나중 사람의 파일이 통째로 남는다
    if mode == "object":
        out = dict(a)
        b_layers = {c[0] for c in b}
        for c in list(out):
            if c[0] in b_layers:  # B가 건드린 레이어는 B 사본의 레이어 전체로 바뀐다
                del out[c]
        out.update(b)
        return out
    if mode == "prop":
        return merge_prop_log(edits)
    if mode == "baton":
        out = dict(a)
        out.update(b)  # B는 A의 결과를 보고 그 위에서 고친다
        return out
    raise ValueError(mode)


def merge_prop_log(edits):
    """속성 단위 LWW, 연산 기록 방식: 서버가 받은 순서대로 칸에 값을 덮어쓴다."""
    out = {}
    for e in sorted(edits, key=lambda e: e["t"]):
        out[e["cell"]] = e["v"]
    return out


def merge_prop_state(edits):
    """속성 단위 LWW, 상태 합치기 방식(대응 구현): 사람마다 칸별 (시각, 값) 레지스터를 들고 있다가
    칸마다 시각이 큰 쪽을 고른다. Wallace(2019)가 비교한 last-writer-wins register와 같은 꼴이다.
    서버가 순서를 정해 주므로 Figma는 타임스탬프가 필요 없다고 썼다. 여기서는 t가 그 순서 역할을 한다."""
    regs = {}
    for who in ("A", "B"):
        mine = {}
        for e in edits:
            if e["who"] == who and (e["cell"] not in mine or e["t"] > mine[e["cell"]][0]):
                mine[e["cell"]] = (e["t"], e["v"])
        for c, r in mine.items():
            if c not in regs or r[0] > regs[c][0]:
                regs[c] = r
    return {c: v for c, (t, v) in regs.items()}


def score(edits, final, informed=False):
    """lost와 silent를 센다(칸 기준, 사람별 마지막 값 기준).
    informed=True(잠금 방식)면 같은 칸 덮어쓰기는 앞사람 결과를 보고 한 것이라 silent 대신 informed로 센다."""
    a, b = _local(edits, "A"), _local(edits, "B")
    lost = silent = 0
    for mine, other in ((a, b), (b, a)):
        for c, v in mine.items():
            if final.get(c) == v:
                continue
            if c in other:
                silent += 1
            else:
                lost += 1
    if informed:
        return {"lost": lost, "silent": 0, "informed": silent}
    return {"lost": lost, "silent": silent}


def baton_wait(edits):
    """잠금 방식에서 B가 기다린 시간(A의 마지막 편집 시각 - B의 첫 편집 시각, 음수면 0).
    둘 다 같은 시간대(0~1)에 일하려 했다는 가정이다."""
    a_end = max(e["t"] for e in edits if e["who"] == "A")
    b_start = min(e["t"] for e in edits if e["who"] == "B")
    return max(0.0, a_end - b_start)


# ---------------------------------------------------------------------------
# 이론값: 칸 N = 레이어 수 × 속성 수, 한 사람이 k번(중복 허용) 고칠 때
# ---------------------------------------------------------------------------

def expected(n_layers, n_props, k):
    """한 사람이 건드린 칸의 기대 개수 = N q,  q = 1 - (1 - 1/N)^k  (칸 하나가 한 번이라도 뽑힐 확률)
    - prop   silent ≈ N q^2      (두 사람이 모두 건드린 칸; 그 칸에서 한쪽 값이 덮인다)
    - file   lost   ≈ N q (1-q)  (먼저 저장한 사람만 건드린 칸이 모두 사라진다)
             silent ≈ N q^2
    - object lost   ≈ N q (r - q),  r = 1 - (1 - 1/L)^k  (B가 그 레이어를 건드렸지만 그 칸은 안 건드림)
    두 사람의 선택이 독립이라는 단순화다. 같은 칸을 두 사람이 서로 다른 값으로 고친 경우만 센다.
    """
    N = n_layers * n_props
    q = 1 - (1 - 1 / N) ** k
    r = 1 - (1 - 1 / n_layers) ** k
    return {
        "touched_per_person": N * q,
        "prop": {"lost": 0.0, "silent": N * q * q},
        "file": {"lost": N * q * (1 - q), "silent": N * q * q},
        "object": {"lost": N * q * (r - q), "silent": N * q * q},
        "baton": {"lost": 0.0, "silent": 0.0},
    }
