"""아주 작은 검색기: 한국어 글자 두 개씩(바이그램) 겹치는 수로 문서를 고른다.

토큰화 선택: 문장부호를 지우고 띄어쓰기로 어절을 나눈 뒤, 어절마다 이웃한 글자 두 개씩을 뽑는다
("환불을" -> "환불", "불을"). 한 글자 어절은 그대로 쓴다. 형태소 분석기 없이 표준 라이브러리만 쓰려고
고른 방식이라, "공짜"와 "무료"처럼 뜻만 같고 글자가 다른 말은 겹치지 않는다(이 장이 보려는 실패).
점수 = 질문과 문서(제목 + 본문)가 함께 가진 서로 다른 바이그램 수. 1등 점수가 0이거나 1등이 여럿이면
(동점) 아무것도 가져오지 않는다고 본다. 동점을 문서 순서로 깨면 운으로 맞히는 경우가 생기기 때문이다.
"""

import re

# \w와 공백이 아닌 문자(문장부호)를 찾는 정규식. 한글은 \w에 포함된다
_PUNCT = re.compile(r"[^\w\s]")


def bigrams(text):
    words = _PUNCT.sub(" ", text).split()
    out = set()
    for w in words:
        # 한 글자 어절은 바이그램을 만들 수 없어 글자 그대로 넣는다
        if len(w) == 1:
            out.add(w)
        # 어절 안에서 이웃한 두 글자. 어절 경계를 넘는 쌍은 만들지 않는다
        for i in range(len(w) - 1):
            out.add(w[i:i + 2])
    return out


def score(query, doc):
    # 점수 = 질문과 문서(제목 + 본문)의 바이그램 집합 교집합 크기. 같은 바이그램은 한 번만 센다
    return len(bigrams(query) & bigrams(doc["title"] + " " + doc["text"]))


def rank(query, docs):
    """[(점수, 문서 id)]를 점수 내림차순, 같은 점수는 id 순으로."""
    return sorted(((score(query, d), d["id"]) for d in docs), key=lambda t: (-t[0], t[1]))


def top1(query, docs):
    r = rank(query, docs)
    s, did = r[0]
    # 1등 점수가 0이거나 1등과 2등이 동점이면 못 찾은 것(None). 동점을 id 순서로 고르면 우연히 맞는 경우가 생긴다
    if s == 0 or (len(r) > 1 and r[1][0] == s):
        return None
    return did


def shared(query, doc):
    return sorted(bigrams(query) & bigrams(doc["title"] + " " + doc["text"]))
