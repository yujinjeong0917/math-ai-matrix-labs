"""모델 평가 9장의 데이터. 손으로 고른 작은 표이거나 고정 시드의 합성 데이터다.

라벨(군집 번호)은 이름표일 뿐이라 번호가 달라도 같은 분할일 수 있다는 점을 보이는 사례들이다.
라벨 배열은 모두 (n,) 정수이고, 점 좌표 X는 (n, 2) 또는 수직선 (n, 1)이다.
"""

import numpy as np


def first_screen():
    """첫 화면 손 계산 예. 학생 6명, 정답 모둠은 (가 가 가 나 나 나).

    perfect: 묶음은 정답과 같고 번호만 엇갈림.  singletons: 모두 한 명씩 따로(아무 의미 없는 결과).
    """
    truth = np.array([0, 0, 0, 1, 1, 1])
    # 묶음은 정답과 같지만 번호가 서로 바뀌어 번호를 그대로 비교한 정확도는 0
    perfect = np.array([1, 1, 1, 0, 0, 0])
    # 모든 점이 따로인 군집(군집 6개). 정보가 없는 결과
    singletons = np.arange(6)
    return truth, perfect, singletons


def fixture_ari_half():
    """eval-ari 카드의 예: 사진 4장, U = (1,2)(3,4), V = (1,3)(2,4). ARI = -1/2."""
    # U = {1,2}{3,4}, V = {1,3}{2,4}. 전체 쌍 6개 중 같이 묶인 쌍이 겹치는 곳이 없어 RI = 1/3, ARI = -1/2
    return np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])


def fixture_nmi():
    """eval-nmi-ami 카드의 예: 정답 (1,1,2,2), 군집 (1,2,3,4). NMI_max = 1/2, NMI_sum = 2/3."""
    # I = log 2(1비트), H(정답) = log 2, H(군집) = log 4. NMI_max = 1/2, NMI_sum = 2I/(H+H) = 2/3
    return np.array([0, 0, 1, 1]), np.array([0, 1, 2, 3])


def fixture_silhouette():
    """eval-silhouette 카드의 예: 수직선 위 0, 2, 10, 12."""
    # 점 0의 실루엣: a = 2, b = (10+12)/2 = 11 -> 9/11
    return np.array([[0.0], [2.0], [10.0], [12.0]])


def vinh_emi_sizes():
    """Vinh·Epps·Bailey(2010) 4.2절의 예: N = 100, 군집 크기 [10]*10 과 [2, 4, 6, 8, 10, 10, 12, 14, 16, 18]. E{I} = 0.4618."""
    # 두 분할의 군집 크기(합이 모두 100). 논문 값 E{I} = 0.4618을 재현하는 입력
    return [10] * 10, [2, 4, 6, 8, 10, 10, 12, 14, 16, 18]


def case_permuted_blobs(seed=0, n_per=100, sigma=0.5):
    """가우스 덩어리 3개(중심 (0,0), (4,0), (2,3.5), 표준편차 0.5), 각 100점.

    반환: X, 정답, 완벽한 군집 결과의 번호를 (0,1,2) -> (2,0,1)로 바꾼 것.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5]])
    # (300, 2) 점. 세 덩어리를 이어 붙인 순서가 y의 순서와 같다
    X = np.concatenate([c + sigma * rng.standard_normal((n_per, 2)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    # relabel[y]로 번호만 바꾼 '완벽한' 군집 결과(묶음은 정답과 같다)
    relabel = np.array([2, 0, 1])
    return X, y, relabel[y]


def case_blobs(k=4, n_per=75, sigma=0.6, seed=0):
    """k값 고르기용 가우스 덩어리 k개. 중심은 반지름 4인 원 위에 고르게."""
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * np.arange(k) / k
    # (k, 2) 반지름 4인 원 위의 k개 중심. 덩어리 간 거리는 k에 따라 달라진다
    centers = 4.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    X = np.concatenate([c + sigma * rng.standard_normal((n_per, 2)) for c in centers])
    return X, np.repeat(np.arange(k), n_per)


def case_random_labels(n=100, k=10, n_true=5, seed=0):
    """정답은 n_true개 모둠에 고르게(각 n / n_true명), 군집 결과는 점마다 k개 중 하나를 아무렇게나.

    Vinh 4절 첫머리 둘째 예의 실험처럼 군집 결과에 k개 번호가 모두 나오도록 다시 뽑는다.
    """
    rng = np.random.default_rng(seed)
    # 정답 모둠: 각 n/n_true명(n이 n_true의 배수라는 가정)
    truth = np.repeat(np.arange(n_true), n // n_true)
    while True:
        pred = rng.integers(0, k, n)
        # 군집 번호 k개가 모두 쓰일 때까지 다시 뽑는다(비어 있는 군집을 허용하지 않기 위해)
        if len(np.unique(pred)) == k:
            return truth, pred


def case_moons(n_per=150, noise=0.08, seed=0):
    """반달 두 개. 위 반달 (cos t, sin t), 아래 반달 (1 - cos t, 0.5 - sin t), t는 0..pi 고르게, 잡음 표준편차 0.08."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n_per)
    top = np.stack([np.cos(t), np.sin(t)], axis=1)
    # 아래 반달: 위 반달을 (1, 0.5)로 옮기고 뒤집은 모양. 두 반달이 맞물린다
    bot = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.concatenate([top, bot]) + noise * rng.standard_normal((2 * n_per, 2))
    return X, np.repeat([0, 1], n_per)


def case_circles(n_per=150, r_in=0.4, noise=0.05, seed=0):
    """중심이 같은 두 원(반지름 1과 0.4), 잡음 표준편차 0.05."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n_per, endpoint=False)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    # 안쪽 원: 바깥 원의 점을 r_in(0.4)배로 줄인다
    inner = r_in * outer
    X = np.concatenate([outer, inner]) + noise * rng.standard_normal((2 * n_per, 2))
    return X, np.repeat([0, 1], n_per)


def case_merge_vs_split(seed=0, sizes=(100, 100, 20, 20, 20, 20), flip=0.1):
    """ARI와 AMI가 엇갈리는지 보는 두 후보.

    정답: 큰 모둠 2개(100명씩) + 작은 모둠 4개(20명씩), 모두 280명.
    A(합치기): 작은 모둠 4개를 하나로 합침. 나머지는 그대로.
    B(쪼개기): 큰 모둠 2개를 각각 반으로 쪼갬. 작은 모둠은 그대로.
    두 후보 모두 점의 flip 비율을 아무 번호로 다시 붙여 잡음을 넣는다.
    """
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(len(sizes)), sizes)
    # A(합치기): 정답 2~5번(작은 모둠 4개)을 2번 하나로
    a = np.where(truth >= 2, 2, truth)
    b = truth.copy()
    # B(쪼개기): 큰 모둠 0, 1번의 절반을 새 번호 6, 7로 옮긴다
    for g, nxt in ((0, 6), (1, 7)):
        idx = np.flatnonzero(truth == g)
        b[idx[rng.permutation(len(idx))[: len(idx) // 2]]] = nxt
    out = []
    for lab in (a, b):
        lab = lab.copy()
        # 점마다 확률 flip(10%)로 라벨을 무작위 번호로 바꿔 잡음을 넣는다
        m = rng.random(len(lab)) < flip
        lab[m] = rng.integers(0, lab.max() + 1, int(m.sum()))
        out.append(lab)
    return truth, out[0], out[1]
