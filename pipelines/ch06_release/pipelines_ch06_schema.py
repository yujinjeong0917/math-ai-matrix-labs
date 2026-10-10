"""6장 되돌리기와 공유 데이터: 앱은 되돌렸는데 데이터는 이미 바뀌었으면?

v1과 v2가 같은 특성 테이블(공유 저장소)을 읽는 블루그린 배포를 흉내 낸다.
v2 배포에 '요금 열을 천 원 단위로 바꾸는' 변경이 함께 들어 있다고 하자.

- in_place:  배포하면서 monthly_fee 열을 그 자리에서 천 원 단위로 덮어쓴다. v1으로 되돌리면 v1은 원 단위를 기대하는데
             천 원 단위를 읽는다. 라우터는 1초 만에 돌아가도 데이터는 돌아가지 않는다.
- expand:    새 열 monthly_fee_k를 추가만 하고 옛 열은 그대로 둔다(넓히기). v2는 새 열, v1은 옛 열을 읽는다.
             v1이 완전히 사라진 뒤에 옛 열을 지운다(좁히기). 되돌려도 v1이 읽는 열은 그대로다.
Fowler의 BlueGreenDeployment(2015 추가 문단)가 말한 '스키마 변경을 앱 업그레이드와 떼어 배포한다'를 작게 옮겼다.
"""

import numpy as np

from pipelines_ch06_model_np import correct


def migrate(shared, mode):
    out = {k: v.copy() for k, v in shared.items()}
    if mode == "in_place":
        out["monthly_fee"] = out["monthly_fee"] / 1000.0
    elif mode == "expand":
        out["monthly_fee_k"] = out["monthly_fee"] / 1000.0
    else:
        raise ValueError(mode)
    return out


def v2_view(shared, mode):
    """v2 코드는 천 원 단위 요금을 기대하므로, 자기 학습 때의 원 단위로 바꿔 읽는다."""
    view = dict(shared)
    fee_k = shared["monthly_fee"] if mode == "in_place" else shared["monthly_fee_k"]
    view["monthly_fee"] = fee_k * 1000.0
    return view


def rollback_accuracy(v1, v2, shared, mode):
    """배포 전 v1, 배포 뒤 v2, 되돌린 뒤 v1의 정확도."""
    before = float(np.mean(correct(v1, shared)))
    migrated = migrate(shared, mode)
    after_v2 = float(np.mean(correct(v2, v2_view(migrated, mode))))
    after_rollback = float(np.mean(correct(v1, migrated)))  # v1은 옛 코드 그대로 monthly_fee를 읽는다
    return {"v1_before": before, "v2_after_deploy": after_v2, "v1_after_rollback": after_rollback}
