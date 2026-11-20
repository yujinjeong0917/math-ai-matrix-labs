"""4장 실습: 지표가 재지 않는 대기 구간을 찾고, 고전 논문 두 편의 숫자를 손계산과 맞춘다.

가상 결제 흐름의 시간은 가상, 예시용이다. 논문 값은 열람한 판에 적힌 것만 쓴다.
Harrison 외(2010): 저자 공개본 PDF를 읽음. Myers(1985): 저자 공개본 PDF를 읽음(표는 스캔이 깨져 쓰지 않음).
사람이 느낀 시간은 계산하지 않는다. 표준 라이브러리만 쓴다.
"""
from __future__ import annotations

# --- 논문 값 ---
# Harrison 외 연구 4: 뒤로 가며 느려지는 줄무늬(리브) 막대와 "같게 느껴진" 단색 막대 길이 조정 결과
HARRISON_SAME_FEEL = {5: 5.61, 15: 16.75}  # 기준 길이(초) -> 조정된 길이(초). 표준오차 0.14, 0.45
HARRISON_REPORTED_PERCENT = {5: 12.2, 15: 11.7}  # 논문이 적은 증가율(%)
# Harrison 외 연구 1 예: 빠르게 증가하는 펄스 대 느리게 감소하는 펄스, 40번 비교
HARRISON_PAIR_EXAMPLE = {"증가 펄스가 빠름": 16, "감소 펄스가 빠름": 6, "같음": 18}

# Myers 직접 비교 설문: 진행 표시를 좋아한 응답 비율(%)과 참여자 수
MYERS_PARTICIPANTS = 48
MYERS_LIKED_PERCENT = 86.1
MYERS_DELAY_RANGE = (1, 17)  # 변동 지연 1~17초 균등분포
MYERS_EMPIRICAL_MEAN = 8.601  # 경험적 평균(초)


# Myers 본문에 적힌 pr 값(논문 표기): 진행 표시 유무, 진행 표시가 있을 때의 변동성 지각
MYERS_PR = {"진행 표시 유무": 0.0006, "변동성 지각(진행 표시 있을 때)": 0.0011, "일정 대 변동(진행 표시 없는 버전)": 0.1497}

# 스스로 점검용 가상 값
QUIZ_FLOW = [("탭부터 로딩 표시까지", 100, "INP"), ("응답 대기", 4000, None)]


def percent_increase(base: float, same_feel: float) -> float:
    return round((same_feel / base - 1) * 100, 1)


def preference_score(first_faster: int, second_faster: int, same: int) -> float:
    """앞 쪽이 빠르다는 응답은 +1, 뒤 쪽이 빠르다는 응답은 -1, 같다는 응답은 0으로 보고 평균을 낸다."""
    n = first_faster + second_faster + same
    return round((first_faster - second_faster) / n, 2)


def uniform_mean(lo: float, hi: float) -> float:
    return (lo + hi) / 2


def denominators_for_percent(percent: float, max_n: int = 48) -> list[tuple[int, int]]:
    """소수 첫째 자리까지 반올림했을 때 percent가 되는 (분자, 분모) 쌍. 분모는 max_n 이하."""
    out = []
    for n in range(1, max_n + 1):
        for k in range(0, n + 1):
            if int(k / n * 1000 + 0.5) / 10 == percent:
                out.append((k, n))
    return out


# --- 가상 결제 흐름 ---
# (구간, 길이 ms, 다루는 지표 또는 None)
FLOW = [
    ("결제하기 탭부터 로딩 표시가 그려질 때까지", 60, "INP"),
    ("로딩 표시부터 결제 결과 화면까지(서버 응답 대기)", 6000, None),
]


def uncovered(flow: list[tuple[str, int, str | None]]) -> tuple[int, int, float]:
    total = sum(ms for _, ms, _ in flow)
    miss = sum(ms for _, ms, who in flow if who is None)
    return miss, total, round(miss / total * 100, 1)


def main() -> None:
    for base, same in HARRISON_SAME_FEEL.items():
        print(f"[Harrison] 기준 {base}초 막대와 같게 느껴진 리브 막대 {same}초: {same}/{base} = {round(same / base, 4)}, 증가율 {percent_increase(base, same)}%(논문 표기 {HARRISON_REPORTED_PERCENT[base]}%), 차이 {round(same - base, 2)}초")
    ex = HARRISON_PAIR_EXAMPLE
    n = sum(ex.values())
    print(f"[Harrison 쌍 비교 예] {ex['증가 펄스가 빠름']}번, {ex['감소 펄스가 빠름']}번, {ex['같음']}번, 합 {n}번: 선호 점수 ({ex['증가 펄스가 빠름']}-{ex['감소 펄스가 빠름']})/{n} = {preference_score(ex['증가 펄스가 빠름'], ex['감소 펄스가 빠름'], ex['같음'])}")
    print(f"[Myers] 변동 지연 {MYERS_DELAY_RANGE[0]}~{MYERS_DELAY_RANGE[1]}초 균등분포의 평균 {uniform_mean(*MYERS_DELAY_RANGE)}초, 논문의 경험적 평균 {MYERS_EMPIRICAL_MEAN}초, 차이 {round(uniform_mean(*MYERS_DELAY_RANGE) - MYERS_EMPIRICAL_MEAN, 3)}초")
    k = MYERS_PARTICIPANTS
    print(f"[Myers] {MYERS_LIKED_PERCENT}% x {k}명 = {round(MYERS_LIKED_PERCENT / 100 * k, 2)}명, 41/{k} = {round(41 / k * 100, 1)}%, 42/{k} = {round(42 / k * 100, 1)}%")
    print(f"[Myers] 분모 48 이하에서 {MYERS_LIKED_PERCENT}%가 되는 (분자, 분모): {denominators_for_percent(MYERS_LIKED_PERCENT)}")
    print(f"[Myers pr] {MYERS_PR}")
    naive = round(6 * HARRISON_REPORTED_PERCENT[15] / 100, 1)
    print(f"[하지 않을 곱셈] 6초 x 11.7% = {naive}초: 5초와 15초에서만 쟀으므로 이 값은 근거가 없다")
    qm, qt, qp = uncovered(QUIZ_FLOW)
    print(f"[점검 3] 탭 {QUIZ_FLOW[0][1]}ms, 대기 {QUIZ_FLOW[1][1]}ms: 세 지표 밖 {qm}/{qt} = {qp}%")
    miss, total, pct = uncovered(FLOW)
    print(f"[결제 흐름] 전체 {total}ms 중 세 지표 밖 {miss}ms = {pct}%")


if __name__ == "__main__":
    main()
