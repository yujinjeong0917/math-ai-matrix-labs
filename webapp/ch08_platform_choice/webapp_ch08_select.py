"""첫 출시 플랫폼 후보를 '조건 충족 목록'으로 내는 선택 도구 (1인 개발 앱 출시 실전 8장).

점수나 순위를 매기지 않는다. 웹, 안드로이드, iOS를 언제나 이 순서로 돌려주고,
플랫폼마다 조건 하나하나에 '충족 / 미충족 / 확인 필요'와 근거(sources 파일의 id)를 붙인다.
'미충족'이 하나도 없는 플랫폼을 '지금 낼 수 있는 후보'라고 부른다. 후보가 여럿이면 고르는 일은 사람 몫이다.

규칙은 2026-10-10에 읽은 공식 문서를 옮긴 것이다(data/sources_2026-10-10.json).
문서가 바뀌면 규칙도 바뀌므로, 결과의 '확인 필요' 칸은 출시 직전에 다시 읽을 목록이다.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
SHARE_DIR = HERE / "data" / "statcounter_2026-10-10"
PLATFORMS = ("web", "android", "ios")  # 고정 순서. 순위가 아니다.
OK, NO, CHECK = "충족", "미충족", "확인 필요"
COUNTRY_FILE = {"KR": "mobile_os_KR.csv", "US": "mobile_os_US.csv", "JP": "mobile_os_JP.csv", "WW": "mobile_os_ww.csv"}


@dataclass
class Target:
    """타깃 조건. 값은 모두 문자열·참거짓이라 표로 적어 두기 쉽다."""
    countries: tuple = ("KR",)          # KR, US, JP, WW(그 밖은 전 세계 값으로 본다)
    pricing: str = "free"               # free, paid_app, subscription
    payment: str = "none"               # none(결제 없음), ads(광고만), digital(앱 안에서 쓰는 디지털 상품), physical(실물·현장 서비스)
    features: tuple = ()                # push, bluetooth, camera
    has_mac: bool = False
    has_windows_pc: bool = True
    has_android_phone: bool = True
    has_iphone: bool = False
    devices_chosen_by_us: bool = False  # 7장 행사장처럼 기기를 우리가 고를 수 있나
    notes: dict = field(default_factory=dict)


def load_share(country):
    """StatCounter 월별 모바일 OS 비율(웹 페이지뷰 기준). 마지막 달 값과 12개월 범위."""
    path = SHARE_DIR / COUNTRY_FILE.get(country, COUNTRY_FILE["WW"])
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    a = [float(r["android"]) for r in rows]
    i = [float(r["ios"]) for r in rows]
    return {"country": country, "months": [rows[0]["month"], rows[-1]["month"]], "last_month": rows[-1]["month"],
            "android_last": a[-1], "ios_last": i[-1],
            "android_avg": round(sum(a) / len(a), 2), "ios_avg": round(sum(i) / len(i), 2),
            "android_range": [min(a), max(a)], "ios_range": [min(i), max(i)]}


def _item(name, status, why, sources):
    return {"item": name, "status": status, "why": why, "sources": list(sources)}


def _web(t):
    items = [_item("설치 없이 주소로 열기", OK, "브라우저만 있으면 열려요. 스토어 심사를 거치지 않아요", [])]
    items.append(_item("개발 장비", OK, "아무 컴퓨터와 브라우저로 만들고 시험할 수 있어요", []))
    if t.payment == "digital":
        items.append(_item("디지털 상품 결제", OK,
                           "PG(결제대행사)와 직접 계약해 웹에서 받아요. 스토어 수수료는 없고 PG 수수료·가입비가 있어요",
                           ["toss_fee"]))
    elif t.payment == "physical":
        items.append(_item("실물·현장 결제", OK, "PG로 받아요", ["toss_fee"]))
    if t.pricing == "paid_app":
        items.append(_item("유료 다운로드", NO, "웹에는 '앱을 사서 내려받는' 단계가 없어요. 구독·이용권으로 바꿔 팔아야 해요", []))
    if "push" in t.features:
        items.append(_item("푸시 알림", CHECK,
                           "안드로이드 크롬은 되고, iPhone은 홈 화면에 추가한 웹앱에서만(iOS 16.4+) 사용자 동작에 이어 권한을 물을 수 있어요",
                           ["webkit_push_164"]))
    if "bluetooth" in t.features:
        st = NO if not t.devices_chosen_by_us else CHECK
        items.append(_item("블루투스 기기 연결", st,
                           "Safari(iOS 포함)는 Web Bluetooth를 지원하지 않아요. 손님 iPhone에서는 안 돼요",
                           ["bcd_compat"]))
    if "camera" in t.features:
        items.append(_item("카메라", CHECK, "브라우저 권한으로 쓸 수 있지만 기종·브라우저마다 시험이 필요해요", []))
    return items


def _android(t):
    items = []
    if t.has_windows_pc or t.has_mac:
        items.append(_item("개발 장비", OK, "Android Studio는 윈도우·맥·리눅스에서 돌아요(ARM 윈도우는 미지원)",
                           ["android_studio_req"]))
    else:
        items.append(_item("개발 장비", NO, "Android Studio를 돌릴 컴퓨터가 없어요", ["android_studio_req"]))
    items.append(_item("실기기 시험", OK if t.has_android_phone else CHECK,
                       "가진 안드로이드 폰으로 시험해요" if t.has_android_phone else "에뮬레이터로 시작하고 실기기를 빌릴 곳을 정해요", []))
    items.append(_item("플레이 개발자 계정", OK, "한 번만 내는 등록비 25달러", ["google_reg_fee"]))
    if t.payment == "digital":
        items.append(_item("디지털 상품 결제", OK,
                           "플레이 결제가 기본이에요. 한국 사용자에게는 대체 결제를 함께 열 수 있고 수수료가 4% 인하돼요",
                           ["gp_payments_policy", "gp_service_fee", "gp_alt_billing_kr", "gp_new_fee_rollout"]))
        if "KR" in t.countries:
            items.append(_item("한국 수수료 구조 변경", CHECK,
                               "2026-12-31부터 한국에도 새 수수료 구조가 적용되는데, 결제 수수료는 아직 발표되지 않았어요",
                               ["gp_new_fee_rollout"]))
        if "US" in t.countries:
            items.append(_item("미국 외부 링크·대체 결제", CHECK,
                               "미국은 2026-06-30부터 새 구조이고, 외부 링크·대체 결제 프로그램도 2026-10-01부터 수수료를 내요",
                               ["gp_service_fee", "gp_us_programs"]))
    elif t.payment == "physical":
        items.append(_item("실물·현장 결제", OK, "실물 상품·현장 서비스에는 플레이 결제를 쓰면 안 되고, PG로 받아요", ["gp_payments_policy"]))
    if t.devices_chosen_by_us:
        items.append(_item("스토어 없이 나눠 주기", CHECK,
                           "7장처럼 APK를 직접 깔 수 있어요. 다만 개발자 인증 요건이 나라별로 적용되기 시작했으니 대상 국가 일정을 확인해요",
                           ["android_dev_verification", "android_unknown_apps"]))
    if "push" in t.features or "bluetooth" in t.features or "camera" in t.features:
        items.append(_item("기기 기능", OK, "네이티브 API 또는 Capacitor 플러그인으로 닿아요", ["capacitor_plugins"]))
    return items


def _ios(t):
    items = []
    if t.has_mac:
        items.append(_item("빌드 장비(맥·Xcode)", OK, "Xcode는 macOS에서 돌아요", ["apple_xcode_req"]))
    else:
        items.append(_item("빌드 장비(맥·Xcode)", NO, "Xcode 시스템 요구사항에 macOS만 적혀 있어 맥이 필요해요(빌려 쓰는 맥·클라우드 맥은 따로 확인)",
                           ["apple_xcode_req", "apple_kr_mac_mini"]))
    items.append(_item("실기기 시험", OK if t.has_iphone else CHECK,
                       "가진 iPhone으로 시험해요" if t.has_iphone else "시뮬레이터로 시작하고 iPhone을 빌릴 곳을 정해요", []))
    items.append(_item("애플 개발자 계정", OK, "연 99달러(나라별 현지 통화로 표시)", ["apple_dev_fee"]))
    items.append(_item("최소 기능 심사(4.2)", CHECK, "웹사이트를 감싸기만 한 앱은 걸릴 수 있어요. 6장 참고", ["apple_42"]))
    if t.payment == "digital":
        items.append(_item("디지털 상품 결제", OK,
                           "앱 안 결제(IAP)가 기본이에요. 나라마다 외부 결제·링크 규칙이 달라요(한국, 미국, EU)",
                           ["apple_sbp", "apple_subs", "apple_eu_terms"]))
        if "KR" in t.countries:
            items.append(_item("한국 외부 결제", CHECK,
                               "제3자 결제를 열면 부가세 포함 금액의 26%에 PG 수수료가 더해져요. 15% 프로그램 회원에게도 26%인지는 문서에 없어요",
                               ["apple_kr_external", "apple_sbp"]))
        if "US" in t.countries:
            items.append(_item("미국 외부 링크 수수료", CHECK,
                               "미국 앱은 외부 링크를 넣을 수 있지만, 링크 구매에 붙을 수수료는 법원 절차가 끝나지 않아 정해지지 않았어요",
                               ["apple_us_links", "epic_apple_9th", "scotus_25_1311"]))
    elif t.payment == "physical":
        items.append(_item("실물·현장 결제", OK, "앱 밖에서 쓰는 실물·서비스는 앱 안 결제가 아닌 방법으로 받아요(3.1.3(e))",
                           ["apple_313e"]))
    if t.pricing == "paid_app":
        items.append(_item("유료 다운로드", OK, "앱 가격에도 같은 수수료율이 붙어요", ["apple_sbp"]))
    if t.devices_chosen_by_us:
        items.append(_item("스토어 없이 나눠 주기", CHECK, "TestFlight·Ad Hoc은 인원·기기 수에 한도가 있어요", ["apple_testflight", "apple_adhoc"]))
    if "push" in t.features or "bluetooth" in t.features or "camera" in t.features:
        items.append(_item("기기 기능", OK, "네이티브 API 또는 Capacitor 플러그인으로 닿아요", ["capacitor_plugins"]))
    return items


RULES = {"web": _web, "android": _android, "ios": _ios}


def evaluate(t):
    """세 플랫폼의 조건 목록. 언제나 PLATFORMS 순서로 돌려준다."""
    out = []
    for p in PLATFORMS:
        items = RULES[p](t)
        counts = {s: sum(1 for i in items if i["status"] == s) for s in (OK, NO, CHECK)}
        out.append({"platform": p, "candidate_now": counts[NO] == 0, "counts": counts, "items": items})
    shares = [load_share(c) for c in t.countries]
    return {"platforms": out, "shares": shares}


def report(result):
    """사람이 읽는 짧은 글. 순서는 고정, 점수 없음."""
    lines = []
    for p in result["platforms"]:
        head = "지금 낼 수 있는 후보" if p["candidate_now"] else "막힌 조건 있음"
        lines.append(f"[{p['platform']}] {head} (충족 {p['counts'][OK]}, 미충족 {p['counts'][NO]}, 확인 필요 {p['counts'][CHECK]})")
        for i in p["items"]:
            lines.append(f"  - {i['status']}: {i['item']} · {i['why']}")
    for s in result["shares"]:
        lines.append(f"참고 {s['country']} 모바일 웹 페이지뷰 {s['last_month']}: 안드로이드 {s['android_last']}%, iOS {s['ios_last']}%"
                     f" (12개월 iOS {s['ios_range'][0]}~{s['ios_range'][1]}%)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(evaluate(Target(payment="digital", pricing="subscription", features=("push",)))))
