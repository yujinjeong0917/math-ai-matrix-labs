"""1인 개발 앱 출시 실전 8장 실험. `uv run python webapp/ch08_platform_choice/experiments.py` 로 실행한다.

E0 모바일 웹 페이지뷰의 OS 비율(StatCounter, 2025-10~2026-09): 마지막 달, 12개월 평균과 범위
E1 결제 한 건 손계산: 월 4,900원 구독 한 건이 경로마다 얼마 남나
E2 월 매출 가정의 수수료 표: 구독자 200명, 1년 넘은 구독자 몫 0%와 40%
E3 첫해 고정비: 계정 비용, 맥이 없을 때 맥 값, 그것을 메울 구독 건수
E4 100만 달러 기준선: 가정 환율로 원화, 지금 매출로 닿는 데 걸리는 달 수
E5 선택 도구: 세 가지 타깃(카드뉴스 편집기 국내 구독, 행사장 주문 앱 손님 폰, 미국 구독)
E6 선택 도구의 순서 불변: 조건을 바꿔도 웹·안드로이드·iOS 순서 그대로

표준 라이브러리만 쓴다. 네트워크 없이 돌고 결과는 매번 같다. 율과 가격은 data/ 의 날짜 붙은 파일에서 읽는다.
"""

import itertools
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import webapp_ch08_fees as F  # noqa: E402
import webapp_ch08_select as S  # noqa: E402

OUT = HERE / "results" / "ch08.json"
PRICE = 4900          # 프로 구독 월 가격(부가세 포함, 이 장의 가정)
SUBSCRIBERS = 200     # 유료 구독자 수(가정)
SCENARIOS = {
    "cardnews_kr_subscription": S.Target(countries=("KR",), pricing="subscription", payment="digital",
                                         features=("push",), has_mac=False, has_windows_pc=True,
                                         has_android_phone=True, has_iphone=False),
    "event_order_guest_phones": S.Target(countries=("KR",), pricing="free", payment="physical",
                                         features=(), has_mac=False, has_windows_pc=True,
                                         has_android_phone=True, has_iphone=False),
    "cardnews_us_subscription_with_mac": S.Target(countries=("US", "KR"), pricing="subscription", payment="digital",
                                                  features=("push",), has_mac=True, has_windows_pc=False,
                                                  has_android_phone=False, has_iphone=True),
}


def e0_shares():
    return {c: S.load_share(c) for c in ("KR", "US", "JP", "WW")}


def e1_one_payment(fees):
    return F.channel_table(PRICE, fees, share_later=0.0)


def e2_monthly(fees):
    gross = PRICE * SUBSCRIBERS
    return {"gross": gross, "price": PRICE, "subscribers": SUBSCRIBERS,
            "share_later_0": F.channel_table(gross, fees, 0.0),
            "share_later_40": F.channel_table(gross, fees, 0.4)}


def e3_fixed(fees, e1):
    net = {r["id"]: r["net_share_of_gross"] for r in e1}
    combos = {"web_only": [], "android": ["android"], "ios_no_mac": ["ios"], "ios_with_mac": ["ios"],
              "android_and_ios_no_mac": ["android", "ios"]}
    out = {}
    for name, plats in combos.items():
        fc = F.fixed_costs(fees, plats, need_new_mac=(name.endswith("no_mac")))
        if name == "web_only":
            pg = fees["pg"]["toss_card_general"]
            fc["items"].append({"item": "PG 가입비 + 연관리비", "krw": pg["signup_krw"] + pg["yearly_krw"], "sources": pg["sources"]})
            fc["first_year_cash"] = sum(i["krw"] for i in fc["items"])
            share = net["web_pg"]
        elif name == "android":
            share = net["gp_15_tier"]
        else:
            share = net["apple_sbp"]
        fc["net_share_used"] = share
        fc["payments_to_cover"] = F.subscribers_to_cover(fc["first_year_cash"], PRICE, share) if fc["first_year_cash"] else 0
        out[name] = fc
    return out


def e4_threshold(fees):
    t = F.threshold_in_krw(fees)
    t["months_at_current_gross"] = round(F.months_to_threshold(PRICE * SUBSCRIBERS, fees), 1)
    return t


def e5_select():
    return {name: S.evaluate(t) for name, t in SCENARIOS.items()}


def e6_order():
    orders = set()
    n = 0
    for mac, iphone, pay, feats in itertools.product((False, True), (False, True),
                                                      ("none", "ads", "digital", "physical"),
                                                      ((), ("push",), ("bluetooth",), ("push", "bluetooth", "camera"))):
        r = S.evaluate(S.Target(has_mac=mac, has_iphone=iphone, payment=pay, features=feats))
        orders.add(tuple(p["platform"] for p in r["platforms"]))
        n += 1
    return {"combinations": n, "distinct_orders": [list(o) for o in orders]}


def main():
    fees = F.load_fees()
    e1 = e1_one_payment(fees)
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "checked": "2026-10-10", "fees_file": F.FEES_FILE.name,
                 "note": "구독 가격·구독자 수·환율·맥 사용 연수는 이 장의 가정이에요. 율은 공식 문서 값이에요."},
        "e0_shares": e0_shares(),
        "e1_one_payment": e1,
        "e2_monthly": e2_monthly(fees),
        "e3_fixed": e3_fixed(fees, e1),
        "e4_threshold": e4_threshold(fees),
        "e5_select": e5_select(),
        "e6_order": e6_order(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("e1_one_payment", "e3_fixed", "e4_threshold")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
