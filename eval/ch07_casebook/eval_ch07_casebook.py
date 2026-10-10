"""모델 평가 7장의 판독 사례집. 지표 표 만들기, 순위 뒤집힘 재기, 짝지은 부트스트랩, 판독 절차, 리포트.

판독 절차(read_table)는 본문 체크리스트를 코드로 옮긴 것이다.
  ① 목적 손실 정하기          (사람이 정한다. 코드는 case["objective"]로 받는다)
  ② 0·작은 값 확인            -> near_zero   : MAPE가 'MAE를 평균 수준으로 나눈 %'의 2배를 넘는 모델이 있다
  ③ 기준선 대비               -> below_mean  : 평가 R^2 < 0인 모델이 있다(평가 데이터 평균으로 찍기보다 못함)
                                 below_naive : MASE > 1인 모델이 있다(학습 구간 1스텝 naive보다 못함)
  ④ 몰림                      -> spiky       : RMSE/MAE가 1.5를 넘는 모델이 있다(정규분포 오차면 약 1.25)
  ⑤ 방향                      -> bias        : |NMBE| > 3%이고 MSE의 절반 넘게가 평균 편향 조각인 모델이 있다
  ⑥ 순위 안정성               -> indistinct  : 짝지은 부트스트랩 95% 구간이 MAE 차이와 RMSE 차이 모두에서 0을 포함한다
문턱값(2, 1.5, 3%, 0.5)은 이 장의 설계 판단이다. 출처가 있는 기준이 아니다.
"""

import json

import numpy as np

import eval_ch07_cases as cases
import eval_ch07_metrics as m

FLAGS = ("near_zero", "below_mean", "below_naive", "spiky", "bias", "indistinct")


def metric_table(case):
    """{'A': {...}, 'B': {...}, 'objective': {...}, 'picks': {지표: 'A'|'B'}, 'objective_pick': 'A'|'B'}"""
    t = {w: m.metric_row(case["y_train"], case["y_test"], case[w]) for w in "AB"}
    obj = {w: cases.objective_loss(case, w) for w in "AB"}
    t["objective"] = obj
    t["objective_pick"] = "A" if obj["A"] < obj["B"] else "B"
    t["picks"] = {k: m.better(k, t["A"][k], t["B"][k]) for k in m.METRICS}
    if "alt_objective" in case:
        alt = {w: cases.objective_loss(case, w, case["alt_objective"]) for w in "AB"}
        t["alt_objective"] = alt
        t["alt_objective_pick"] = "A" if alt["A"] < alt["B"] else "B"
    return t


def _metric_on_index(name, case, which, idx):
    y, p = case["y_test"][idx], case[which][idx]
    if name == "mase":
        return m.mase(case["y_train"], y, p)  # 분모(학습 구간)는 고정
    return getattr(m, name)(y, p)


def paired_bootstrap(case, metrics=m.METRICS, n_boot=1000, seed=0):
    """평가 구간의 같은 위치를 두 모델에 똑같이 뽑아(짝지어) 지표 차이 A - B의 분포를 만든다.
    위치를 서로 독립으로 뽑으므로 시계열의 이웃끼리 닮은 성질은 무시한다(블록 부트스트랩은 범위 밖)."""
    n = len(case["y_test"])
    rng = np.random.default_rng(90_000 + seed)
    idx_all = rng.integers(0, n, size=(n_boot, n))
    scale = m.naive_mae_in_sample(case["y_train"])
    diffs = {k: [] for k in metrics}
    for start in range(0, n_boot, 100):  # 메모리를 아끼려고 100개씩 나눠 계산한다(결과는 같다)
        idx = idx_all[start:start + 100]
        y, a, b = case["y_test"][idx], case["A"][idx], case["B"][idx]
        ea, eb = y - a, y - b
        for k in metrics:
            if k == "mae":
                d = np.abs(ea).mean(1) - np.abs(eb).mean(1)
            elif k == "rmse":
                d = np.sqrt((ea ** 2).mean(1)) - np.sqrt((eb ** 2).mean(1))
            elif k == "r2":
                sst = ((y - y.mean(1, keepdims=True)) ** 2).sum(1)
                d = -((ea ** 2).sum(1) - (eb ** 2).sum(1)) / sst
            elif k == "mape":
                d = 100 * (np.abs(ea) / np.abs(y)).mean(1) - 100 * (np.abs(eb) / np.abs(y)).mean(1)
            elif k == "mase":
                d = (np.abs(ea).mean(1) - np.abs(eb).mean(1)) / scale
            elif k == "nmbe":
                d = np.abs(100 * ea.sum(1) / (n * y.mean(1))) - np.abs(100 * eb.sum(1) / (n * y.mean(1)))
            else:
                raise ValueError(k)
            diffs[k].append(d)
    out = {}
    for k in metrics:
        d = np.concatenate(diffs[k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        va, vb = _metric_on_index(k, case, "A", slice(None)), _metric_on_index(k, case, "B", slice(None))
        full = abs(va) - abs(vb) if k == "nmbe" else va - vb
        out[k] = {"diff": float(full), "lo": float(lo), "hi": float(hi), "contains_zero": bool(lo <= 0.0 <= hi)}
    return out


def read_table(table, boot=None):
    """지표 표(와 부트스트랩)만 보고 판독 깃발을 켠다. 목적 손실은 보지 않는다."""
    A, B = table["A"], table["B"]
    f = {
        "near_zero": max(A["mape_inflation"], B["mape_inflation"]) > 2.0,
        "below_mean": min(A["r2"], B["r2"]) < 0.0,
        "below_naive": max(A["mase"], B["mase"]) > 1.0,
        "spiky": max(A["rmse_mae_ratio"], B["rmse_mae_ratio"]) > 1.5,
        "bias": any(abs(r["nmbe"]) > 3.0 and r["bias_share"] > 0.5 for r in (A, B)),
        "indistinct": bool(boot is not None and boot["mae"]["contains_zero"] and boot["rmse"]["contains_zero"]),
    }
    return {k: bool(v) for k, v in f.items()}


def ranking_flip_rate(case_fn, metric, n_seeds=100, objective=None):
    """시드 n_seeds개에서 지표 하나로 고른 모델이 목적 손실로 고른 모델과 다른 비율."""
    miss = 0
    for s in range(n_seeds):
        c = case_fn(s)
        t = metric_table(c)
        target = t["objective_pick"] if objective is None else (
            "A" if cases.objective_loss(c, "A", objective) < cases.objective_loss(c, "B", objective) else "B")
        miss += t["picks"][metric] != target
    return miss / n_seeds


def _round(x, nd=6):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    return x


def dumps(obj):
    """같은 입력이면 바이트 단위로 같은 문자열. 키 정렬, 소수 6자리."""
    return json.dumps(_round(obj), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def render_report(res):
    """results/report.md 본문. 지표 표 하나만 받은 독자가 무엇을 보게 되는지 사례마다 한 줄씩."""
    lines = ["# 모델 평가 7장 판독 리포트", "", "숫자는 results/ch07.json(시드 0)에서 옮겼다. 깃발은 read_table의 판독 결과다.", ""]
    for name in cases.CASE_NAMES:
        e = res["E1"][name]
        A, B = e["table"]["A"], e["table"]["B"]
        on = [k for k, v in e["flags"].items() if v]
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| | MAE | RMSE | R2 | MAPE | MASE | NMBE | 목적 손실 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for w, r in (("A", A), ("B", B)):
            lines.append(f"| {w} | {r['mae']:.3f} | {r['rmse']:.3f} | {r['r2']:.3f} | {r['mape']:.2f}% | {r['mase']:.3f} | "
                         f"{r['nmbe']:.2f}% | {e['table']['objective'][w]:.3f} |")
        lines.append("")
        lines.append(f"- 목적 손실({e['objective']})이 고른 모델: {e['table']['objective_pick']}")
        lines.append(f"- 지표별 선택: " + ", ".join(f"{k}={v}" for k, v in e["table"]["picks"].items()))
        lines.append(f"- 켜진 깃발: {', '.join(on) if on else '없음'}")
        lines.append("")
    return "\n".join(lines)
