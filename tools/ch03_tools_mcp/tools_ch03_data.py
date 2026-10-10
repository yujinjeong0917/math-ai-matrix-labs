"""AI 도구 실전 3장: 관통 프로젝트의 문의 데이터(가상).

2장의 "이번 주 문의 120건"을 실제 파일로 만든다. 유형 9가지와 건수는 이 장에서 정한 가정값이고,
문장은 틀에 번호를 끼운 가짜 문의다. 고정 시드라 몇 번을 만들어도 같은 파일이 나온다.

    uv run python tools/ch03_tools_mcp/tools_ch03_data.py   # data/inquiries_2026-W41.json 다시 만들기
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
DATA_FILE = HERE / "data" / "inquiries_2026-W41.json"
WEEK = "2026-W41"
WEEK_START = datetime(2026, 10, 5, 9, 0)   # ISO 2026년 41주는 10월 5일(월)부터 11일(일)까지
SEED = 3

# 2장 보고서 절 12개 가운데 앞의 9개가 문의 유형이다. 건수 합은 120.
COUNTS = {"배송": 28, "환불": 18, "결제 오류": 14, "회원 정보": 10, "쿠폰": 12,
          "앱 오류": 13, "상품 문의": 12, "교환": 8, "칭찬": 5}
CATEGORIES = list(COUNTS)

TEMPLATES = {
    "배송": ["주문 {o} 배송이 {d}일째 그대로예요.", "{o} 송장 번호가 조회되지 않아요.", "{o} 받는 주소를 바꾸고 싶어요."],
    "환불": ["{o} 환불이 아직 안 들어왔어요.", "{o} 부분 환불이 되나요?", "{o} 취소했는데 카드 승인이 남아 있어요."],
    "결제 오류": ["결제 버튼을 누르면 오류 코드 {e}가 떠요.", "{o} 결제가 두 번 됐어요.", "간편결제가 {e} 오류로 멈춰요."],
    "회원 정보": ["비밀번호 재설정 메일이 안 와요.", "휴대폰 번호를 바꾸고 싶어요.", "탈퇴하면 적립금은 어떻게 되나요?"],
    "쿠폰": ["{c} 쿠폰이 적용되지 않아요.", "{c} 쿠폰 기간이 지났는데 연장되나요?", "첫 구매 쿠폰이 안 보여요."],
    "앱 오류": ["앱이 장바구니에서 꺼져요(버전 {v}).", "앱 로그인이 계속 풀려요(버전 {v}).", "앱 알림이 두 번씩 와요."],
    "상품 문의": ["{p} 재입고 언제 되나요?", "{p} 크기가 어떻게 되나요?", "{p} 세탁기로 빨아도 되나요?"],
    "교환": ["{o} 사이즈 교환하고 싶어요.", "{o} 색상이 사진과 달라 교환 원해요.", "{o} 교환 회수가 안 왔어요."],
    "칭찬": ["배송이 빨라서 좋았어요.", "상담원 답변이 친절했어요.", "포장이 꼼꼼했어요."],
}
CHANNELS = ["앱", "웹", "전화", "메일"]


def build(seed=SEED):
    rng = random.Random(seed)
    cats = [c for c in CATEGORIES for _ in range(COUNTS[c])]
    rng.shuffle(cats)
    minutes = sorted(rng.sample(range(0, 7 * 24 * 60 - 9 * 60), len(cats)))
    items = []
    for i, (c, m) in enumerate(zip(cats, minutes), start=1):
        t = WEEK_START + timedelta(minutes=m)
        text = rng.choice(TEMPLATES[c]).format(
            o=f"A{rng.randint(10000, 99999)}", d=rng.randint(3, 9), e=f"P-{rng.randint(100, 999)}",
            c=rng.choice(["가을맞이", "10% 할인", "무료배송"]), v=f"5.{rng.randint(0, 4)}.{rng.randint(0, 9)}",
            p=rng.choice(["린넨 셔츠", "울 니트", "캔버스 가방"]))
        items.append({"id": f"#{i:04d}", "received": t.strftime("%Y-%m-%dT%H:%M"),
                      "channel": rng.choice(CHANNELS), "category": c, "text": text})
    return {"_about": "관통 프로젝트의 가상 고객 문의 한 주 분량. 유형별 건수와 문장은 이 장에서 정한 가정값이다. "
                      "tools_ch03_data.py가 고정 시드로 만든다.",
            "week": WEEK, "categories": CATEGORIES, "items": items}


def load(path=DATA_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    DATA_FILE.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(DATA_FILE)


if __name__ == "__main__":
    main()
