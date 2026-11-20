"""1장 실습: 한 사용자의 32건 기록에서 이해와는 별개로 불확실성 인지와 신뢰 보정을 계산한다 (가상 데이터).

모든 숫자는 가상의 예시이다. 표준 라이브러리만 쓴다.

가상 과제 32건: AI가 높은 확신(확률 0.7 초과)으로 답한 16건과 낮은 확신 16건.
칸별 건수(가상): 높은 확신 중 맞음 14, 틀림 2 / 낮은 확신 중 맞음 8, 틀림 8. 즉 AI 정확도는 22/32.
(확률 0.7을 기준으로 가른 것은 Wang & Yin(2022)의 과제 설계를 따라 한 것이고, 이 장의 가상 숫자는 그 논문의 값이 아니다.)

사용자 한 명의 기록은 칸별로 "몇 건 따랐나"로 적는다.
불확실성 인지(이 장의 정의) = 높은 확신 때 의존율 - 낮은 확신 때 의존율
"""

from __future__ import annotations

CELLS = {  # (확신, 정답) -> 건수
    ("높음", "맞음"): 14,
    ("높음", "틀림"): 2,
    ("낮음", "맞음"): 8,
    ("낮음", "틀림"): 8,
}

USERS = {  # 칸별로 따른 건수(가상)
    "P": {("높음", "맞음"): 13, ("높음", "틀림"): 2, ("낮음", "맞음"): 4, ("낮음", "틀림"): 3},  # 확신을 보고 가림
    "Q": {("높음", "맞음"): 14, ("높음", "틀림"): 2, ("낮음", "맞음"): 8, ("낮음", "틀림"): 8},  # 전부 따름
    "R": {("높음", "맞음"): 7, ("높음", "틀림"): 1, ("낮음", "맞음"): 4, ("낮음", "틀림"): 4},   # 어디서나 절반만 따름
}


def check(followed: dict) -> None:
    for k, v in followed.items():
        if not (0 <= v <= CELLS[k]):
            raise ValueError(f"{k}: 따른 건수 {v}가 칸 크기 {CELLS[k]}를 벗어났다")


def measures(followed: dict) -> dict:
    check(followed)
    n = sum(CELLS.values())
    n_high = sum(v for (c, _), v in CELLS.items() if c == "높음")
    n_low = n - n_high
    f_high = sum(v for (c, _), v in followed.items() if c == "높음")
    f_low = sum(v for (c, _), v in followed.items() if c == "낮음")
    n_right = sum(v for (_, a), v in CELLS.items() if a == "맞음")
    n_wrong = n - n_right
    fr = sum(v for (_, a), v in followed.items() if a == "맞음")
    fw = sum(v for (_, a), v in followed.items() if a == "틀림")
    return {
        "reliance_high": f_high / n_high,
        "reliance_low": f_low / n_low,
        "awareness_gap": f_high / n_high - f_low / n_low,
        "appropriate": (fr + (n_wrong - fw)) / n,
        "appropriate_count": fr + (n_wrong - fw),
        "overtrust": fw / n_wrong,
        "undertrust": (n_right - fr) / n_right,
    }


def ai_accuracy() -> float:
    right = sum(v for (_, a), v in CELLS.items() if a == "맞음")
    return right / sum(CELLS.values())


def main() -> None:
    print(f"[1] 가상 과제 {sum(CELLS.values())}건, AI 정확도 {ai_accuracy() * 100:.2f}% "
          f"(높은 확신 {sum(v for (c, _), v in CELLS.items() if c == '높음')}건, 낮은 확신 16건)")
    for name, f in USERS.items():
        m = measures(f)
        print(f"[2] 사용자 {name}: 높은 확신 의존 {m['reliance_high'] * 100:.2f}%, 낮은 확신 의존 {m['reliance_low'] * 100:.2f}%, "
              f"차이 {m['awareness_gap'] * 100:.2f}%p, 적정 신뢰 {m['appropriate_count']}/32={m['appropriate'] * 100:.2f}%, "
              f"과신 {m['overtrust'] * 100:.1f}%, 과소신뢰 {m['undertrust'] * 100:.1f}%")


if __name__ == "__main__":
    main()
