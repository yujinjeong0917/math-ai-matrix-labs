"""월 매출 가정에서 스토어 수수료와 웹 결제(PG) 수수료를 비교하는 계산기 (1인 개발 앱 출시 실전 8장).

바뀌는 값(수수료율, 계정 비용, 장비 가격, 환율 가정)은 코드에 적지 않고 data/fees_2026-10-10.json 에서 읽는다.
값이 바뀌면 날짜가 다른 새 파일을 만들고 FEES_FILE 만 바꾼다.

계산의 기준(이 장에서 정한 가정):
- 고객이 낸 금액 P 는 부가세 10%가 포함된 값이다. 부가세 P/11 은 누가 걷든 개발자 몫이 아니라고 보고 먼저 뗀다.
- 스토어 수수료는 부가세를 뗀 금액(공급가액)에 매긴다(애플 Schedule 2의 'net of any and all taxes').
  예외로 애플 한국 외부 결제 26%는 문서가 'gross of any value-added taxes'라고 적어 고객이 낸 금액 전체에 매긴다(store_on="gross").
- PG 수수료는 카드사·PG 관행대로 고객이 낸 금액 P 전체에 매긴다고 둔다. 수수료에 붙는 부가세는 넣지 않는다.
- 외부 결제(앱 밖 결제 링크, 제3자 결제)는 줄어든 스토어 수수료와 PG 수수료를 둘 다 낸다.
- 율을 확인하지 못한 경로는 0으로 채우지 않고 None 으로 두어 결과에도 '확인 필요'로 남긴다.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
FEES_FILE = HERE / "data" / "fees_2026-10-10.json"


def load_fees(path=FEES_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def blended_store_rate(first_year_rate, later_rate, share_later):
    """구독 매출 중 share_later 만큼이 '유료 1년 넘은 구독자'에게서 나올 때의 평균 수수료율.

    애플의 구독 15%는 앱 단위가 아니라 구독자 한 사람의 유료 기간 단위로 정해져서 섞어 계산한다.
    """
    if not 0 <= share_later <= 1:
        raise ValueError("share_later 는 0~1")
    return (1 - share_later) * first_year_rate + share_later * later_rate


def split(gross, store_rate, pg_rate, vat_rate=0.10, store_on="base"):
    """고객이 낸 금액 gross(원)를 부가세, 스토어 수수료, PG 수수료, 개발자 몫으로 나눈다.

    store_rate 나 pg_rate 가 None 이면(확인 못 한 율) 개발자 몫도 None 이다.
    """
    vat = gross * vat_rate / (1 + vat_rate)
    base = gross - vat
    if store_rate is None or pg_rate is None:
        return {"gross": gross, "vat": round(vat), "base": round(base), "store_fee": None, "pg_fee": None,
                "net": None, "net_share_of_gross": None, "unverified": True}
    store_fee = (gross if store_on == "gross" else base) * store_rate
    pg_fee = gross * pg_rate
    net = base - store_fee - pg_fee
    return {"gross": gross, "vat": round(vat), "base": round(base), "store_fee": round(store_fee),
            "pg_fee": round(pg_fee), "net": round(net), "net_share_of_gross": round(net / gross, 4),
            "unverified": False}


def channel_table(gross, fees, share_later=0.0):
    """fees['channels'] 의 결제 경로마다 split 을 돌린다. 경로 순서는 파일에 적힌 순서 그대로다(순위 아님)."""
    rows = []
    for ch in fees["channels"]:
        rate = ch["store_rate"]
        if ch.get("subscription_later_rate") is not None and rate is not None:
            rate = blended_store_rate(rate, ch["subscription_later_rate"], share_later)
        pg = fees["pg"][ch["pg"]]["rate"] if ch["pg"] else 0.0
        r = split(gross, rate, pg, fees["assumptions"]["vat_rate"], ch.get("store_on", "base"))
        r.update({"id": ch["id"], "label": ch["label"], "store_rate_used": None if rate is None else round(rate, 4),
                  "pg_rate_used": pg, "sources": ch["sources"]})
        rows.append(r)
    return rows


def threshold_in_krw(fees):
    """소규모 사업자 프로그램·첫 100만 달러 기준선을 환율 가정으로 원화로 바꾼다."""
    usd = fees["assumptions"]["threshold_usd"]
    fx = fees["assumptions"]["krw_per_usd"]
    return {"usd": usd, "krw_per_usd": fx, "krw": usd * fx}


def months_to_threshold(monthly_gross, fees):
    """월 매출이 그대로일 때 기준선(부가세 뗀 금액 기준)에 닿는 데 걸리는 달 수."""
    t = threshold_in_krw(fees)["krw"]
    base = monthly_gross / (1 + fees["assumptions"]["vat_rate"])
    return t / base


def fixed_costs(fees, platforms, need_new_mac):
    """첫해 고정비(원). 개발자 계정 비용과, 맥이 없을 때 새로 살 맥 값을 더한다.

    맥 값은 상각하지 않은 '첫해에 나가는 현금'이다. 연 단위로 나눈 값은 따로 돌려준다.
    """
    acc = fees["accounts"]
    out = {"items": [], "first_year_cash": 0, "per_year_mac_spread": 0}
    if "ios" in platforms:
        a = acc["apple"]
        out["items"].append({"item": a["label"], "krw": round(a["usd"] * fees["assumptions"]["krw_per_usd"]),
                             "sources": a["sources"]})
        if need_new_mac:
            mac = fees["mac"]
            out["items"].append({"item": mac["label"], "krw": mac["krw"], "sources": mac["sources"]})
            out["per_year_mac_spread"] = round(mac["krw"] / fees["assumptions"]["mac_years"])
    if "android" in platforms:
        g = acc["google"]
        out["items"].append({"item": g["label"], "krw": round(g["usd"] * fees["assumptions"]["krw_per_usd"]),
                             "sources": g["sources"]})
    out["first_year_cash"] = sum(i["krw"] for i in out["items"])
    return out


def subscribers_to_cover(cost_krw, price_gross, net_share):
    """한 달 net_share(고객이 낸 금액 대비 개발자 몫)일 때 cost_krw 를 메우려면 몇 건의 결제가 필요한가(올림)."""
    per = price_gross * net_share
    n = cost_krw / per
    return int(n) if n == int(n) else int(n) + 1
