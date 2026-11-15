"""모델 평가 7장 검증. `uv run pytest -q eval/ch07_casebook` 로 실행한다."""

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_ch07_cases as cases  # noqa: E402
import eval_ch07_casebook as cb  # noqa: E402
import eval_ch07_metrics as m  # noqa: E402
import eval_ch07_torch as tm  # noqa: E402

# 장마다 experiments.py가 있어 'import experiments'는 pytest에서 다른 장 모듈과 섞일 수 있다. 고유한 이름으로 읽는다.
import importlib.util  # noqa: E402

# 파일 경로로 직접 읽어 고유 모듈 이름(eval_ch07_experiments)을 붙인다
_spec = importlib.util.spec_from_file_location("eval_ch07_experiments", Path(__file__).parent / "experiments.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)

N = 30  # 테스트용 시드 수(실험은 100개)


def test_first_screen_hand_numbers():
    a, b = cases.first_screen()
    # 오차가 모두 양수라 평균이 곧 MAE. 쿠폰(10분 넘으면 5,000원)은 A만 한 번 받고, 분당 100원 할인은 A 1,500원 < B 2,000원
    assert float(np.mean(a)) == 3.0 and float(np.mean(b)) == 4.0          # MAE: 15/5, 20/5
    assert float(np.sqrt(np.mean(a ** 2))) == 5.0                           # RMSE: sqrt(125/5)
    assert float(np.sqrt(np.mean(b ** 2))) == 4.0                           # RMSE: sqrt(80/5)
    e0 = ex.e0_first_screen()
    assert (e0["A"]["coupon_won"], e0["B"]["coupon_won"]) == (5000, 0)
    assert (e0["A"]["per_minute_won"], e0["B"]["per_minute_won"]) == (1500, 2000)


# fixture 손 계산: NMBE(B) = 100*8/(4*15) = 40/3. 부동소수를 분수로 되돌려 비교
def test_fixture_exact():
    ytr, yte, a, b = cases.fixture_small()
    assert m.naive_mae_in_sample(ytr) == 5 / 3
    ra, rb = m.metric_row(ytr, yte, a), m.metric_row(ytr, yte, b)
    assert (ra["mae"], ra["rmse"], rb["mae"], rb["rmse"]) == (1.0, 1.0, 2.0, 2.0)
    assert abs(ra["r2"] - 0.96) < 1e-12 and abs(rb["r2"] - 0.84) < 1e-12
    assert abs(ra["mape"] - 7.5) < 1e-12 and abs(rb["mape"] - 15.0) < 1e-12
    assert abs(ra["mase"] - 0.6) < 1e-12 and abs(rb["mase"] - 1.2) < 1e-12
    assert ra["nmbe"] == 0.0
    assert Fraction(rb["nmbe"]).limit_denominator(100) == Fraction(40, 3)  # 100*8/(4*15)


def test_sign_convention_underprediction_is_positive():
    y = np.array([10.0, 20.0, 30.0])
    assert m.nmbe(y, y - 1.0) > 0 > m.nmbe(y, y + 1.0)


# 일곱 사례 모두에서 여섯 지표가 NumPy와 torch에서 같다(상대·절대 허용 1e-10)
def test_numpy_matches_torch_on_every_case():
    for name in cases.CASE_NAMES:
        c = cases.MAKERS[name](3)
        for w in "AB":
            nps = m.metric_row(c["y_train"], c["y_test"], c[w])
            ts = tm.metrics(torch.tensor(c["y_train"]), torch.tensor(c["y_test"]), torch.tensor(c[w]))
            for k in m.METRICS:
                assert math.isclose(nps[k], ts[k], rel_tol=1e-10, abs_tol=1e-10), (name, w, k)


# N개 시드 중 조건 pred를 만족한 비율. 합성 데이터라 시드마다 조금씩 달라서 비율로 확인
def _share(name, pred):
    return sum(bool(pred(cases.MAKERS[name](s))) for s in range(N)) / N


def _t(c):
    return cb.metric_table(c)


# 사례마다 의도한 엇갈림이 시드의 90% 이상에서 나타나는지(설계한 구조가 우연이 아님을 확인)
def test_intended_disagreements_appear_in_most_seeds():
    assert _share("spiky", lambda c: (lambda t: t["picks"]["mae"] == "A" and t["picks"]["rmse"] == "B")(_t(c))) >= 0.9
    assert _share("small_values", lambda c: (lambda t: t["picks"]["r2"] == "A" and t["picks"]["mape"] == "B")(_t(c))) >= 0.9
    assert _share("flat_level", lambda c: (lambda t: t["A"]["mape"] < 1.0 and t["A"]["mase"] > 1.0)(_t(c))) >= 0.9
    assert _share("monthly_bias", lambda c: (lambda t: t["picks"]["rmse"] == "B" and t["picks"]["nmbe"] == "A")(_t(c))) >= 0.9
    assert _share("fixable_bias", lambda c: (lambda t: t["A"]["nmbe"] > 5.0 and t["A"]["cv_rmse"] < 15.0)(_t(c))) >= 0.9
    assert _share("extrapolation", lambda c: _t(c)["A"]["r2"] < 0.0) >= 0.9


# 판독 깃발이 의도한 사례에서 90% 이상 켜지는지. 대조군의 indistinct는 부트스트랩이 필요해 아래 테스트가 따로 본다
def test_intended_flag_fires_in_most_seeds():
    for name in cases.CASE_NAMES:
        flag = cases.MAKERS[name](0)["intended_flag"]
        if flag == "indistinct":
            continue
        hits = sum(cb.read_table(cb.metric_table(cases.MAKERS[name](s)))[flag] for s in range(N))
        assert hits / N >= 0.9, (name, flag, hits)


def test_control_is_read_as_indistinct():
    """대조군은 실험과 같은 설정(시드 100개, 부트스트랩 1000번)으로 잰다.
    95% 구간은 지표마다 약 5%는 0을 빼고 그려지므로(두 지표를 같이 보면 그보다 조금 더) 90%가 아니라 85%를 기준으로 둔다."""
    hits = 0
    for s in range(100):
        c = cases.make_case_control(s)
        boot = cb.paired_bootstrap(c, metrics=("mae", "rmse"), n_boot=1000, seed=s)
        hits += cb.read_table(cb.metric_table(c), boot)["indistinct"]
    assert hits >= 85, hits


# 같은 표라도 목적이 바뀌면 답이 바뀐다: 큰 실수 비용 기준은 B, 단순 절대오차 기준은 A
def test_objective_changes_the_answer():
    t = cb.metric_table(cases.make_case_spiky(0))
    assert t["objective_pick"] == "B" and t["alt_objective_pick"] == "A"


def test_same_test_set_rank_equivalence():
    """같은 평가 데이터에서는 R^2와 RMSE, MASE와 MAE의 순위가 늘 같다(분모가 두 모델에 같아서)."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        ytr, y = rng.normal(50, 5, 30), rng.normal(50, 5, 40)
        a, b = y + rng.normal(0, rng.uniform(1, 5), 40), y + rng.normal(rng.normal(), rng.uniform(1, 5), 40)
        ra, rb = m.metric_row(ytr, y, a), m.metric_row(ytr, y, b)
        assert m.better("r2", ra["r2"], rb["r2"]) == m.better("rmse", ra["rmse"], rb["rmse"])
        assert m.better("mase", ra["mase"], rb["mase"]) == m.better("mae", ra["mae"], rb["mae"])


def test_bootstrap_is_paired_and_chunk_safe():
    c = cases.make_case_control(0)
    b1 = cb.paired_bootstrap(c, n_boot=250, seed=5)
    b2 = cb.paired_bootstrap(c, n_boot=250, seed=5)
    # 같은 시드면 재현. n_boot=250은 100개씩 나누는 청크가 마지막에 50개로 끝나는 경우를 시험
    assert b1 == b2
    # 두 모델을 같게 두면 차이가 정확히 0이어야 한다(같은 위치를 뽑았으니까)
    # B를 A와 같게 만든 사본. 짝지은 재표집이면 모든 재표집에서 차이가 정확히 0
    same = dict(c, B=c["A"])
    for v in cb.paired_bootstrap(same, n_boot=200, seed=0).values():
        assert v["lo"] == v["hi"] == v["diff"] == 0.0


# 시간 값이 없어 두 번 만든 JSON 문자열과 리포트가 완전히 같아야 한다
def test_report_is_byte_identical_for_same_seed():
    part = lambda: {"E0": ex.e0_first_screen(), "E1": ex.e1_seed0(), "E7": ex.e7_fixture_torch()}  # noqa: E731
    s1, s2 = cb.dumps(part()), cb.dumps(part())
    assert s1 == s2
    r = json.loads(s1)
    assert cb.render_report(r) == cb.render_report(json.loads(s2))


def test_results_json_schema():
    p = Path(__file__).parent / "results" / "ch07.json"
    r = json.loads(p.read_text())
    assert set(r) == {"env", "E0", "E1", "E2", "E2_alt", "E3", "E4", "E5", "E6", "E7"}
    assert set(r["E1"]) == set(cases.CASE_NAMES)
    for v in r["E1"].values():
        assert set(v["table"]["picks"]) == set(m.METRICS)
        assert set(v["flags"]) == set(cb.FLAGS)
    # 저장된 결과 파일 기준: 같은 평가 데이터에서 R^2-RMSE, MASE-MAE 순위 불일치는 0건
    assert r["E5"]["r2_vs_rmse_disagree"] == 0 and r["E5"]["mase_vs_mae_disagree"] == 0
