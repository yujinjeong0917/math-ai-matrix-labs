"""시드 10개 리포트. results/ch11.json만 읽어 report/seeds_report.html(외부 파일 없는 한 장)을 만든다.
`uv run python rl/ch11_trpo_ppo/report/seeds_report.py`

설정마다: 시드 10개의 학습 곡선(가는 선)과 평균(굵은 선), 최종 리턴의 평균과 부트스트랩 95% 구간,
시드 {0..4}와 {5..9}로 나눈 두 묶음의 평균 곡선, Welch t 검정(양측), 126가지 나눔 중 p < 0.05인 수.
"""

import html
import json
from pathlib import Path

HERE = Path(__file__).parent
DATA = json.loads((HERE.parent / "results" / "ch11.json").read_text())
SHOW = ["pg_lr10", "pg_lr100", "pg_lr1000", "trpo", "trpo_no_linesearch", "ppo_eps0.1", "ppo_eps0.2", "ppo_eps0.3",
        "ppo_noclip", "ppo_eps0.2_adam0.2", "ppo_noclip_adam0.2"]
W, H, PAD = 320, 150, 26


def path(ys, n):
    pts = [f"{PAD + (W - 2 * PAD) * i / (n - 1):.1f},{H - PAD - (H - 2 * PAD) * y / 200:.1f}" for i, y in enumerate(ys)]
    return "M" + " L".join(pts)


def mean(cols):
    return [sum(c) / len(c) for c in zip(*cols)]


def svg(curves):
    n = len(curves[0])
    thin = "".join(f'<path d="{path(c, n)}" class="seed"/>' for c in curves)
    a, b = mean(curves[:5]), mean(curves[5:])
    axes = (f'<line x1="{PAD}" y1="{H - PAD}" x2="{W - PAD}" y2="{H - PAD}" class="ax"/>'
            f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H - PAD}" class="ax"/>'
            f'<text x="{PAD - 4}" y="{PAD + 4}" class="lab" text-anchor="end">200</text>'
            f'<text x="{PAD - 4}" y="{H - PAD + 4}" class="lab" text-anchor="end">0</text>'
            f'<text x="{W - PAD}" y="{H - 8}" class="lab" text-anchor="end">갱신 {n}번</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="시드별 학습 곡선">{axes}{thin}'
            f'<path d="{path(a, n)}" class="ga"/><path d="{path(b, n)}" class="gb"/></svg>')


def card(name):
    s = {**DATA["E1_E2_E3_summary"], **DATA["E3_adam_lr_sweep"]}[name]
    sp = DATA["E4_seed_split"][name]
    t = sp["split_0_4_vs_5_9"]
    return f"""<section class="card"><h2>{html.escape(name)}</h2>{svg(DATA['curves_per_seed'][name])}
<p>최종 리턴 평균 <b>{s['final_mean']:.1f}</b>, 95% 구간 {s['final_ci95'][0]:.1f} ~ {s['final_ci95'][1]:.1f}, 최저 시드 {s['final_min']:.1f}</p>
<p><span class="ka">시드 0~4</span> {t['mean_a']:.1f} vs <span class="kb">시드 5~9</span> {t['mean_b']:.1f}, Welch t = {t['t']:.2f}, 양측 p = {t['p_two_sided']:.3f}</p>
<p>126가지 5+5 나눔 중 p &lt; 0.05: {sp['splits_p_lt_0.05']}개 · 붕괴 기준에 걸린 시드: {len(s['collapsed_seeds'])}개 · 최종 20 미만: {s['seeds_final_below_20']}개</p></section>"""


def main():
    cards = "\n".join(card(n) for n in SHOW)
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>시드 10개 리포트 · 11장</title><style>
:root {{ --paper:#fbfaf7; --ink:#1d1d1b; --dim:#66645e; --line:#d8d5cc; --a:#2f6fb0; --b:#c0602a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --paper:#161614; --ink:#ecebe6; --dim:#a19f97; --line:#3a3935; --a:#7fb2e5; --b:#eb9a6b; }} }}
body {{ margin:0; background:var(--paper); color:var(--ink); font-family:system-ui, sans-serif; line-height:1.6; }}
main {{ max-width:1040px; margin:0 auto; padding:32px 16px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:18px; }}
.card {{ background:color-mix(in srgb, var(--paper) 92%, var(--ink)); border-radius:12px; padding:12px 14px; }}
.card h2 {{ font-size:15px; margin:0 0 6px; font-family:ui-monospace, monospace; }}
.card p {{ font-size:13px; margin:4px 0; color:var(--dim); }} .card b {{ color:var(--ink); }}
svg {{ width:100%; height:auto; }} .seed {{ fill:none; stroke:var(--dim); stroke-width:.6; opacity:.45; }}
.ga {{ fill:none; stroke:var(--a); stroke-width:2; }} .gb {{ fill:none; stroke:var(--b); stroke-width:2; }}
.ax {{ stroke:var(--line); }} .lab {{ font-size:9px; fill:var(--dim); }} .ka {{ color:var(--a); }} .kb {{ color:var(--b); }}
</style></head><body><main><h1>시드 10개 리포트</h1>
<p>cart-pole(최대 200걸음), 갱신 60번, 갱신마다 16판. 가는 선은 시드 하나, 파란 선은 시드 0~4의 평균, 주황 선은 시드 5~9의 평균이에요.
최종 리턴은 마지막 10번 갱신의 평균이고, 구간은 시드 10개를 1만 번 다시 뽑은 백분위 부트스트랩이에요.</p>
<div class="grid">{cards}</div></main></body></html>"""
    (HERE / "seeds_report.html").write_text(page)
    print(HERE / "seeds_report.html")


if __name__ == "__main__":
    main()
