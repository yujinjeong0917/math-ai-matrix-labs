"""모델 평가 3장의 데이터. Anscombe 네 쌍은 원 논문 표 그대로, 나머지는 합성 데이터.

anscombe()는 {이름: (x, y)}(각 (11,)), first_screen()은 (x_train, y_train, x_test, y_test),
case_offset/case_scaled는 (y, yhat), case_extrapolation은 (x_train, y_train, x_test, y_test)를 돌려준다.
R^2가 높아도 모델이 맞다고 할 수 없는 경우(같은 R^2의 다른 데이터, 오프셋, 외삽)를 만드는 용도다.
"""

import numpy as np

# Anscombe, F. J. (1973). Graphs in Statistical Analysis. The American Statistician 27(1):17-21, p.19 TABLE.
# "Four data sets, each comprising 11 (x, y) pairs." 1~3번은 x가 같아 원문에서도 한 번만 적었다.
# 네 쌍 모두 x 평균 9, y 평균 약 7.5, 회귀선 y = 3 + 0.5x, R^2 약 0.667로 요약 통계가 같다(테스트가 확인)
_X123 = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0]
_Y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
_Y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
_Y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
_X4 = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 19.0, 8.0, 8.0, 8.0]
_Y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]


def anscombe():
    """{'1': (x, y), ..., '4': (x, y)}. 원문 표의 관측 순서 그대로."""
    return {
        "1": (np.array(_X123), np.array(_Y1)),
        "2": (np.array(_X123), np.array(_Y2)),
        "3": (np.array(_X123), np.array(_Y3)),
        "4": (np.array(_X4), np.array(_Y4)),
    }


def first_screen():
    """첫 화면 손 계산 예. 학습 x = 0..4, 테스트 x = 5..8, 정답 y = x^2 (잡음 없음)."""
    # 학습 x = 0,1,2,3,4 (y = 0,1,4,9,16). 곡선인데 직선을 맞추고 바깥 구간 5..8에서 잰다
    xtr = np.arange(5.0)
    xte = np.arange(5.0, 9.0)
    return xtr, xtr**2, xte, xte**2


def case_offset(n=200, shift=5.0, seed=0):
    """정답 y ~ N(0, 1), 예측 yhat = y + shift. 함께 움직이지만(r^2 = 1) 늘 shift만큼 위에 있다."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 1.0, n)
    # y와 완전히 같이 움직이므로 상관계수 제곱 r^2 = 1. 그러나 오차는 매번 +shift라 SSE = n shift^2
    return y, y + shift


def case_scaled(n=200, scale=2.0, seed=0):
    """정답 y ~ N(0, 1), 예측 yhat = scale * y. 역시 r^2 = 1인데 크기가 틀렸다."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 1.0, n)
    # 선형 변환이라 r^2 = 1. 크기만 틀린 모델
    return y, scale * y


def quad_sample(lo, hi, n, rng, noise=1.0):
    """x ~ U(lo, hi), y = x^2 + N(0, noise^2)."""
    x = rng.uniform(lo, hi, n)
    # 참 함수는 2차. 직선 모델은 이 곡률을 놓친다
    return x, x**2 + rng.normal(0.0, noise, n)


def case_extrapolation(seed=0, n_train=50, n_test=50, train=(0.0, 5.0), test=(5.0, 10.0), noise=1.0):
    """[0, 5]에서 y = x^2 + 잡음을 뽑아 직선을 맞추고, test 구간에서 잰다."""
    rng = np.random.default_rng(seed)
    # 같은 rng로 학습과 테스트를 연달아 뽑아 시드 하나로 전체가 재현된다
    xtr, ytr = quad_sample(*train, n_train, rng, noise)
    xte, yte = quad_sample(*test, n_test, rng, noise)
    return xtr, ytr, xte, yte
