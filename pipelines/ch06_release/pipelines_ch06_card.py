"""6장 모델 카드: 배포 승인 전에 사람이 읽는 한 장짜리 설명서를 만든다.

Mitchell 등(2018, arXiv:1810.03993)의 모델 카드처럼 성능을 숫자 하나가 아니라 구간(여기서는 지역)별로 나눠 적는다.
이 장에서 보여 주려는 것: 오프라인 검증 데이터로 만든 카드는 서빙에서만 생기는 문제(부산 앱의 단위)를 보여 주지 못한다.
그래서 카드에는 '이 숫자가 어디서 왔는지'와 '카나리에서 다시 볼 것'을 함께 적는다.
"""

import numpy as np

from pipelines_ch06_data import LABEL, REGIONS
from pipelines_ch06_model_np import correct


def by_region(model, table):
    c = correct(model, table)
    rows = {"all": {"n": int(len(c)), "acc": float(c.mean())}}
    for r, name in enumerate(REGIONS):
        m = table["region"] == r
        rows[name] = {"n": int(m.sum()), "acc": float(c[m].mean())}
    return rows


def render(name, v1_rows, v2_rows, source, numeric):
    lines = [
        f"# 모델 카드: {name}",
        "",
        "## 쓰임새",
        "- 고객 이탈 가능성을 0/1로 예측해 상담 우선순위를 정한다. 요금이나 계약 조건을 자동으로 바꾸는 데는 쓰지 않는다.",
        "",
        "## 입력",
        f"- 수치 열: {', '.join(numeric)}",
        "- 범주 열: plan(3), region(4)",
        "- usage_drop은 비율(0~1)이어야 한다. 퍼센트(0~100)로 들어오면 결과를 믿을 수 없다.",
        "",
        f"## 오프라인 정확도 (출처: {source})",
        "",
        "| 구간 | 요청 수 | v1 | v2 | 차이 |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in v1_rows:
        a, b = v1_rows[k]["acc"], v2_rows[k]["acc"]
        lines.append(f"| {k} | {v1_rows[k]['n']} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    lines += [
        "",
        "## 알려진 한계",
        "- 위 숫자는 정리된 오프라인 로그에서 왔다. 앱마다 서빙 입력을 만드는 코드는 검증하지 않았다.",
        "- 정답(이탈 여부)은 늦게 들어온다. 카나리에서는 정답이 들어온 요청만 센다.",
        "",
        "## 배포 조건",
        "- 카나리 5%, 전체와 지역별로 v1 대비 오류율 z > 3이면 되돌린다. v2 요청 400개를 넘기면 승인 요청.",
        "- 승인: ml-release-owners",
        "",
    ]
    return "\n".join(lines)


def label_rate(table):
    return float(np.mean(table[LABEL]))
