"""모델 평가 9장의 PyTorch 대응 구현.

이 실습 저장소는 numpy·torch·pytest만 쓰므로(scikit-learn 없음), 설계도의 sklearn 대조 대신
같은 식을 텐서 연산으로 다시 써서 NumPy 구현과 같은 입력에서 값을 대조한다.
대응표는 one-hot 행렬곱으로, E{I}는 torch.lgamma로, 실루엣 거리는 torch.cdist로 계산한다.
"""

import torch

torch.set_num_threads(2)
DT = torch.float64


def _onehot(labels):
    # 값을 0..k-1 번호로 재매김(NumPy의 _codes와 같음)
    _, inv = torch.unique(torch.as_tensor(labels), return_inverse=True)
    return torch.nn.functional.one_hot(inv).to(DT)


# one-hot 행렬곱으로 대응표를 만든다: (R, n) @ (n, C) = (R, C)
def contingency(u, v):
    return _onehot(u).T @ _onehot(v)  # (R, C)


def _c2(x):
    return x * (x - 1) / 2


def ari(u, v):
    t = contingency(u, v)
    n = t.sum()
    s_both = _c2(t).sum()
    s_u = _c2(t.sum(1)).sum()
    s_v = _c2(t.sum(0)).sum()
    # 순열 모형에서 같은 쌍 수의 기댓값(NumPy의 E[RI] 경로와 다른 식이지만 ARI 값은 같다)
    exp = s_u * s_v / _c2(n)
    # ARI = (같은 쌍 수 - 기대) / ((U 같은 쌍 수 + V 같은 쌍 수)/2 - 기대). 분모 0인 퇴화 경우는 처리하지 않는다(NumPy판과 다름)
    return ((s_both - exp) / ((s_u + s_v) / 2 - exp)).item()


def mutual_info(u, v):
    t = contingency(u, v)
    n = t.sum()
    outer = t.sum(1, keepdim=True) @ t.sum(0, keepdim=True)
    m = t > 0
    return (t[m] / n * torch.log(n * t[m] / outer[m])).sum().item()


def entropy(labels):
    c = _onehot(labels).sum(0)
    p = c / c.sum()
    return -(p * torch.log(p)).sum().item()


def expected_mutual_info(u, v):
    t = contingency(u, v)
    a, b = t.sum(1), t.sum(0)
    n = t.sum()
    lf = lambda x: torch.lgamma(x + 1)  # noqa: E731  log(x!)
    nmax = int(torch.minimum(a.max(), b.max()).item())
    # n_ij 후보를 마지막 축에 놓고 (R, C, K)로 한 번에 계산
    nij = torch.arange(1, nmax + 1, dtype=DT)[None, None, :]  # (1, 1, K)
    A, B = a[:, None, None], b[None, :, None]
    # 각 (i, j)마다 초기하분포의 지지 구간에 속하는 n_ij만 유효
    valid = (nij >= A + B - n) & (nij <= torch.minimum(A, B))
    # 유효하지 않은 칸을 1로 바꿔 lgamma에 음수가 들어가지 않게 하고, 마지막에 term을 0으로 마스킹
    nij_c = torch.where(valid, nij, torch.ones_like(nij))
    logp = (lf(A) + lf(B) + lf(n - A) + lf(n - B) - lf(n)
            - lf(nij_c) - lf(A - nij_c) - lf(B - nij_c) - lf(n - A - B + nij_c))
    term = nij_c / n * torch.log(n * nij_c / (A * B)) * torch.exp(logp)
    return torch.where(valid, term, torch.zeros_like(term)).sum().item()


def ami(u, v, average="max"):
    hu, hv = entropy(u), entropy(v)
    # 두 정규화만 지원(대조 검증용)
    d = {"max": max(hu, hv), "arithmetic": (hu + hv) / 2}[average]
    i, e = mutual_info(u, v), expected_mutual_info(u, v)
    return (i - e) / (d - e)


def silhouette(X, labels, direct=True):
    """direct=False는 torch.cdist 기본 설정(점이 많으면 행렬곱 꼴로 거리 계산). 반올림 오차 비교용."""
    X = torch.as_tensor(X, dtype=DT)
    oh = _onehot(labels)  # (n, k)
    mode = "donot_use_mm_for_euclid_dist" if direct else "use_mm_for_euclid_dist_if_necessary"
    # 점이 많을 때 cdist 기본값은 행렬곱 꼴(|x|^2+|y|^2-2xy)이라 반올림 오차가 커진다. 그래서 직접 계산을 기본으로 둔다
    D = torch.cdist(X, X, compute_mode=mode)  # direct: NumPy와 같은 직접 계산
    sums = D @ oh  # (n, k): 군집마다 거리 합
    size = oh.sum(0)
    own = oh @ size
    # 자기 군집 거리 합 / (크기 - 1). 크기 1이면 분모 0을 피하려 1로 올리고 아래에서 s를 0으로 둔다
    a = (sums * oh).sum(1) / (own - 1).clamp(min=1)
    mean_other = (sums / size).masked_fill(oh.bool(), float("inf"))
    b = mean_other.min(1).values
    s = torch.where(own > 1, (b - a) / torch.maximum(a, b), torch.zeros_like(a))
    return s.mean().item()
