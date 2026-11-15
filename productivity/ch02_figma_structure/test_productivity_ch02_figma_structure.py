"""생산성 도구 2장 검증. `uv run pytest -q productivity/ch02_figma_structure` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
"""

import json
import random
from pathlib import Path

import productivity_ch02_layout as L
import productivity_ch02_project as P
import productivity_ch02_sync as S

HERE = Path(__file__).parent


def _hug(text_w, pad=10):
    return {"name": "F", "kind": "frame", "layout": "H", "pad": (pad, pad, 0, 0), "gap": 0, "sx": "hug", "sy": "hug",
            "children": [{"name": "t", "kind": "text", "text": "x", "sx": "fixed", "sy": "hug", "w": text_w}]}


def test_help_doc_hug_example():
    # Figma 도움말: 글자 40px + 좌우 여백 10px → 60px, 글자 50px → 70px (hug 계산이 도움말 예시와 일치하는지)
    assert L.layout(_hug(40))["box"][2] == 60
    assert L.layout(_hug(50))["box"][2] == 70


def test_first_screen_hand_numbers():
    # 버튼: 여백 16 + 라벨 "예약"(14×2=28) + 16 = 60, 라벨 "예약하기"(56)면 88
    assert L.text_width("예약") == 28 and L.text_width("예약하기") == 56
    assert 16 + 28 + 16 == 60 and 16 + 56 + 16 == 88
    # 430px 화면에서 좌우 여백 16을 지키려면 상단바 폭은 430 - 32 = 398, 고정 좌표면 343 그대로라 오른쪽 여백 71
    assert 430 - 2 * 16 == 398 and 430 - 16 - 343 == 71
    assert P.buttons_in(P.spec_file()) == 7  # 목록 4 + 상세·예약·완료 각 1(손계산은 모형 글자 폭 상수에서 나온다)


def test_fill_child_turns_hug_parent_fixed():
    # 자식 중 fill이 있으면 hug 부모가 그 축에서 fixed(폭 200)로 취급된다. fill 자식은 200 - 여백 20 - 글자 40 = 130을 받는다
    f = _hug(40)
    f["w"] = 200
    f["children"].append({"name": "spacer", "kind": "text", "text": "", "sx": "fill", "sy": "hug"})
    assert L.effective_sizing(f, "w") == "fixed"
    laid = L.layout(f)
    assert laid["box"][2] == 200
    assert laid["children"][1]["box"][2] == 200 - 10 - 40 - 10


def test_min_max_clamp():
    # hug 폭 60이 min_w 80보다 작아 80으로 올라가고, 폭 420이 max_w 120보다 커서 120으로 잘린다
    f = _hug(40)
    f["min_w"] = 80
    assert L.layout(f)["box"][2] == 80
    f = _hug(400)
    f["max_w"] = 120
    assert L.layout(f)["box"][2] == 120


def test_layout_matches_closed_form_on_random_stacks():
    # 임의의 가로 쌓기 500개에 대해 레이아웃 계산이 닫힌 식 "여백 + 자식 폭 합 + 간격 x (개수-1) + 여백"과 같은지 본다
    rng = random.Random(3)
    for _ in range(500):
        n = rng.randint(1, 6)
        kids = [P.text(f"t{j}", "가" * rng.randint(0, 6)) for j in range(n)]
        pl, pr, g = rng.randrange(0, 33, 4), rng.randrange(0, 33, 4), rng.randrange(0, 33, 4)
        f = {"name": "F", "kind": "frame", "layout": "H", "pad": (pl, pr, 0, 0), "gap": g, "sx": "hug", "sy": "hug", "children": kids}
        assert L.layout(f)["box"][2] == pl + sum(L.text_width(k["text"]) for k in kids) + g * (n - 1) + pr


def test_frozen_file_looks_identical_at_375():
    # 375px에서는 두 파일의 모든 상자(box)가 같아야 한다: 차이는 구조뿐이고 겉모양은 같다
    spec = P.resolved_screens(P.spec_file())
    pp = P.resolved_screens(P.pingpong_file())
    for a, b in zip(spec, pp):
        la, lb = L.layout(a, 0, 0, 375, 812), L.layout(b, 0, 0, 375, 812)
        boxes_a = [n["box"] for n, _ in L.walk(la)]
        boxes_b = [n["box"] for n, _ in L.walk(lb)]
        assert boxes_a == boxes_b


def test_width_change_fixed_vs_auto():
    pp, spec = P.lint(P.pingpong_file()), P.lint(P.spec_file())
    assert spec["width_checks"]["430"] == {"overflow": 0, "stretch_fail": 0}
    # 13곳 = 화면 4개의 stretch 의도 요소 5(목록: 상단바 + 줄 4) + 2 + 4 + 2. 고정 좌표 판은 430px에서 우측 여백 71px로 어긋난다
    assert pp["width_checks"]["430"]["stretch_fail"] == 13
    assert pp["width_checks"]["375"] == {"overflow": 0, "stretch_fail": 0}


def test_component_propagation_override_and_detach():
    # 갱신 5곳 = 목록 버튼 4 + 상세 CTA 1. 완료 화면은 fill을 override해서 남고(kept_by_override), 예약 화면은 detach해서 놓친다
    doc = P.spec_file(overrides_drift=True, detach_drift=True)
    P.change_main(doc, {"fill": "#0B7A55"})
    rep = P.stale_report(doc, {"fill": "#0B7A55"})
    assert rep["updated"] == 5
    assert [s for s, _ in rep["kept_by_override"]] == ["S04_완료_기본"]
    assert [s for s, _ in rep["missed_by_detach"]] == ["S03_예약_기본"]
    # CTA 문구는 override라서 main 문구를 바꿔도 남는다
    doc = P.spec_file()
    P.change_main(doc, {"label.text": "예약하기"})
    labels = [n["text"] for s in P.resolved_screens(doc) for n, _ in L.walk(s) if n.get("name") == "label"]
    assert labels.count("예약하기") == 5 and "예약 확정" in labels and "목록으로" in labels


def test_lint_counts():
    assert P.lint(P.spec_file())["frame_name_violations"] == 0
    assert P.lint(P.pingpong_file())["frame_name_violations"] == 4  # 핑퐁 파일의 화면 이름 4개가 모두 규칙 밖
    assert P.lint(P.spec_file(detach_drift=True))["detached_instances"] == 1
    assert P.lint(P.spec_file())["spacing_off_token"] == 0


def test_prop_lww_two_implementations_agree():
    # 속성 단위 병합을 "연산 기록 순서대로 덮어쓰기"와 "칸별 시각 레지스터 비교" 두 방식으로 구현해 1000가지 입력에서 같은 결과인지 대조한다
    rng = random.Random(11)
    for i in range(1000):
        e = S.make_edits(rng.randint(1, 15), rng.randint(1, 5), rng.randint(1, 25), seed=i)
        assert S.merge_prop_log(e) == S.merge_prop_state(e)


def test_merge_invariants():
    for seed in range(300):
        e = S.make_edits(10, 4, 12, seed)
        a, b = S._local(e, "A"), S._local(e, "B")
        both = set(a) & set(b)  # 두 사람이 모두 고친 칸. 속성 단위 병합의 silent 수가 정확히 이 칸 수와 같다
        assert S.score(e, S.merge(e, "prop"))["lost"] == 0
        assert S.score(e, S.merge(e, "prop"))["silent"] == len(both)
        assert S.score(e, S.merge(e, "file")) == {"lost": len(set(a) - set(b)), "silent": len(both)}
        assert S.score(e, S.merge(e, "baton"), informed=True)["silent"] == 0
        assert S.score(e, S.merge(e, "object"))["lost"] <= S.score(e, S.merge(e, "file"))["lost"]


def test_theory_matches_simulation():
    # 이론식(expected)과 시뮬레이션 평균을 상대 오차로 비교한다. 허용폭: 파일 5%, 레이어 10%, silent 15%(+0.1은 값이 작을 때의 여유). 변수 이름의 밑줄은 모듈 별칭 L, P와 겹치지 않게 한 것
    L_, P_, k, T = 40, 6, 20, 400
    th = S.expected(L_, P_, k)
    m = {mode: 0.0 for mode in ("file", "object", "prop")}
    s = 0.0
    for seed in range(T):
        e = S.make_edits(L_, P_, k, seed=50_000 + seed)
        for mode in m:
            m[mode] += S.score(e, S.merge(e, mode))["lost"] / T
        s += S.score(e, S.merge(e, "prop"))["silent"] / T
    assert abs(m["file"] - th["file"]["lost"]) < 0.05 * th["file"]["lost"]
    assert abs(m["object"] - th["object"]["lost"]) < 0.1 * th["object"]["lost"]
    assert abs(s - th["prop"]["silent"]) < 0.15 * th["prop"]["silent"] + 0.1
    assert m["prop"] == 0


def test_variant_combinatorics_from_help_doc():
    # Figma 도움말: 체크박스 5개를 모든 조합으로 프로토타입하면 프레임 32개, 연결 160개
    assert 2**5 == 32 and 5 * 2**5 == 160


def test_results_file_matches_modules():
    # results/ch02.json은 experiments.py를 돌려 만든 파일이다. 파일이 없으면 이 테스트는 실패한다(1장 테스트와 달리 건너뛰지 않는다)
    res = json.loads((HERE / "results" / "ch02.json").read_text())
    assert res["e4_lint"]["pingpong"] == P.lint(P.pingpong_file())
    assert res["e4_lint"]["spec"] == P.lint(P.spec_file())
    assert res["e5_adoption"]["spec"] == P.adoption_score(P.spec_file())
    assert res["e1_min_failure"]["copy_file_save_B_last"] == {"lost": 7, "silent": 0}
    assert res["e7_cross_check"]["prop_lww_log_vs_state_mismatch"] == 0
