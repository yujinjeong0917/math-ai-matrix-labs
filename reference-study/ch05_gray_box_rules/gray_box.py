"""5장: 회색 상자 점검와 규칙 옮기기를 숫자로 점검한다. (제안된 연습법이며 효과가 입증된 기준이 아니다.)

가상 화면, 예시용: 도서관 대출 앱의 홈 화면 상자 다섯 개를 가정한다. 좌표와 크기는 모두 가상이다.
"""

# (이름, 너비, 높이, 의도한 중요도 순위: 1이 가장 중요)
BOXES = [
    ("대출 중인 책", 320, 160, 1),
    ("검색창", 320, 48, 2),
    ("추천 목록", 320, 120, 3),
    ("공지 띠", 320, 32, 4),
    ("하단 메뉴", 320, 56, 5),
]


def areas(boxes):
    return [(n, w * h, r) for n, w, h, r in boxes]


def spearman(x, y):
    """순위 상관계수: 순위의 피어슨 상관."""
    rx, ry = _plain_ranks(x), _plain_ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None  # 한쪽 순위가 모두 같으면(모든 값이 같으면) 상관을 계산할 수 없다
    return cov / (vx * vy) ** 0.5


def _plain_ranks(values):
    """작은 값이 1위(오름차순), 같은 값은 평균 순위."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def gray_box_check(boxes):
    """면적이 가장 큰 상자가 의도한 1순위인지, 면적 순위와 의도한 순위가 얼마나 같은지."""
    a = areas(boxes)
    biggest = max(a, key=lambda t: t[1])[0]
    intended_first = next(n for n, _, r in a if r == 1)
    # 의도 순위 1이 가장 중요 -> 면적은 클수록 중요: 면적과 (-의도순위)를 비교
    rho = spearman([t[1] for t in a], [-t[2] for t in a])
    return biggest == intended_first, rho


def ratio_chain(sizes):
    """큰 값부터 정렬한 크기에서 이웃한 비율."""
    s = sorted(sizes, reverse=True)
    return [a / b for a, b in zip(s, s[1:])]


def ratio_gap(ref, new):
    """두 크기 목록의 이웃 비율 차이의 평균 절댓값."""
    r, n = ratio_chain(ref), ratio_chain(new)
    return sum(abs(a - b) for a, b in zip(r, n)) / len(r)


def shared_values(ref, new):
    """겹치는 절대값 개수(베낀 값)."""
    return sorted(set(ref) & set(new))


def area_ranks(boxes):
    """면적이 큰 상자가 1위."""
    order = sorted(boxes, key=lambda b: -b[1] * b[2])
    return {b[0]: i + 1 for i, b in enumerate(order)}


def rank_gaps(boxes):
    """상자마다 (이름, 의도한 순위, 면적 순위, 칸 차이 d)."""
    ar = area_ranks(boxes)
    return [(n, r, ar[n], ar[n] - r) for n, w, h, r in boxes]


def squared_gap_sum(boxes):
    return sum(d * d for _, _, _, d in rank_gaps(boxes))


def rho_by_height(boxes):
    """너비를 무시하고 높이만으로 순위를 매겼을 때의 순위 상관(비교용)."""
    return spearman([h for _, _, h, _ in boxes], [-r for _, _, _, r in boxes])


def total_ratio(sizes):
    return max(sizes) / min(sizes)


# 이 사이트의 다른 화면(학습 트랙 목록 화면, tracks/index.html)에 홈의 원칙을 옮겨 보는 확인값. 1440px 폭.
# 홈 .hero { padding-top: 96px }, 트랙 목록 화면 인라인 CSS .tracks-hero { padding-top: 56px }.
# 두 화면의 h1은 모두 class="t-hero"(--fs-hero)라서 같은 토큰을 쓴다. 본문 글자 --fs-body는 17px.
HOME_HERO_PADDING_TOP = 96
TRACKS_HERO_PADDING_TOP = 56
H1_1440 = 56
BODY = 17


REF_SIZES_1440 = [56, 24, 17]  # 이 사이트 홈, 1440px 폭: h1, 리드, 본문
NEW_SIZES = [44, 20, 15]       # 가상의 새 화면, 예시용

if __name__ == "__main__":
    ok, rho = gray_box_check(BOXES)
    print("면적:", [(n, w * h) for n, w, h, _ in BOXES])
    print(f"가장 큰 상자가 의도한 1순위인가: {ok}, 순위 상관 {rho:.3f}")
    print("  순위 차이 d(면적 순위 - 의도 순위):", [(n, d) for n, _, _, d in rank_gaps(BOXES)], "제곱합", squared_gap_sum(BOXES))
    # 하나를 고쳐서 위계를 흐리는 경우
    broken = [(n, w, 140 if n == "공지 띠" else h, r) for n, w, h, r in BOXES]
    ok2, rho2 = gray_box_check(broken)
    print("공지 띠를 140px 높이로 키우면 면적", [w * h for n, w, h, _ in broken if n == "공지 띠"][0],
          f"(본래 면적의 {[w * h for n, w, h, _ in broken if n == '공지 띠'][0] / 10240:.1f}배)")
    print(f"  1순위 일치 {ok2}, 순위 상관 {rho2:.3f}")
    print("  순위 차이 d(면적 순위 - 의도 순위):", [(n, d) for n, _, _, d in rank_gaps(broken)], "제곱합", squared_gap_sum(broken))
    # 모든 상자를 같은 크기로 만들면
    same = [(n, 320, 80, r) for n, w, h, r in BOXES]
    print("상자를 모두 320x80으로 같게 만들면 순위 상관:", gray_box_check(same)[1], "(면적이 모두 같아 계산할 수 없음)")
    # 너비가 다른 상자가 있으면
    narrow = [(n, 120 if n == "추천 목록" else w, h, r) for n, w, h, r in BOXES]
    print("추천 목록 너비를 120px로 줄이면 면적", [w * h for n, w, h, _ in narrow if n == "추천 목록"][0])
    print(f"추천 목록 너비를 120px로 줄이면: 면적 순위 상관 {gray_box_check(narrow)[1]:.3f}, 높이만 본 순위 상관 {rho_by_height(narrow):.3f}")
    print(f"  (너비가 모두 같을 때: 면적 순위 상관 {rho:.3f}, 높이만 본 순위 상관 {rho_by_height(BOXES):.3f})")
    print("세 값의 비율(기준):", [round(x, 4) for x in ratio_chain(REF_SIZES_1440)])
    print("세 값의 비율(새 화면):", [round(x, 4) for x in ratio_chain(NEW_SIZES)])
    rr, nn = ratio_chain(REF_SIZES_1440), ratio_chain(NEW_SIZES)
    print("세 값의 비율 차이(절대값):", [round(abs(x - y), 4) for x, y in zip(rr, nn)])
    print(f"비율 차이 평균 {ratio_gap(REF_SIZES_1440, NEW_SIZES):.4f}, 겹치는 절대값 {shared_values(REF_SIZES_1440, NEW_SIZES)}")
    print(f"전체 비율(가장 큰 / 가장 작은): 기준 {total_ratio(REF_SIZES_1440):.3f}, 새 화면 {total_ratio(NEW_SIZES):.3f},"
          f" 차이 {total_ratio(REF_SIZES_1440) - total_ratio(NEW_SIZES):.3f}")
    print(f"[트랙 목록 화면에 옮겨 보기, 폭 1440px] 제목/본문(--fs-body 17px): 홈 {H1_1440 / BODY:.3f}, 트랙 목록 화면 {H1_1440 / BODY:.3f} (같은 --fs-hero 토큰)")
    print(f"  위쪽 여백: 홈 {HOME_HERO_PADDING_TOP}px, 트랙 목록 화면 {TRACKS_HERO_PADDING_TOP}px (비 {TRACKS_HERO_PADDING_TOP / HOME_HERO_PADDING_TOP:.3f})")
    print(f"  제목 크기 대비 위쪽 여백: 홈 {HOME_HERO_PADDING_TOP / H1_1440:.3f}배, 트랙 목록 화면 {TRACKS_HERO_PADDING_TOP / H1_1440:.3f}배")
