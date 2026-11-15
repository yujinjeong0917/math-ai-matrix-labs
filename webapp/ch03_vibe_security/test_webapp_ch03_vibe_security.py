"""1인 개발 앱 출시 실전 3장 검증. `uv run pytest -q webapp/ch03_vibe_security` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
서버는 127.0.0.1의 빈 포트에 잠깐 띄운다. 바깥 네트워크와 API 키 없이 돈다.
"""

import json
import math
from pathlib import Path
from urllib.parse import quote

import pytest

import webapp_ch03_app as A
import webapp_ch03_ratelimit as R
import webapp_ch03_rls as L
import webapp_ch03_upload as U

HERE = Path(__file__).parent
ALICE, BOB, ADMIN = "demo-session-alice", "demo-session-bob", "demo-session-admin"


# BEFORE(모두 꺼짐)에서 시작해 시험할 장치만 켜는 헬퍼.
def fixes(**kw):
    f = dict(A.BEFORE)
    f.update(kw)
    return f


def read_cards(base, token):
    code, _, body = A.request(base, "/rest/v1/cards", token=token, apikey=A.PUBLISHABLE_KEY)
    assert code == 200
    return [r["id"] for r in json.loads(body)]


# ---------- 1. RLS ----------

def test_rls_off_everyone_reads_everything():
    with A.running(fixes(rls="off")) as (base, _):
        assert read_cards(base, None) == [1, 2, 3, 4, 5]
        assert read_cards(base, ALICE) == [1, 2, 3, 4, 5]


def test_rls_on_without_policy_is_default_deny():
    with A.running(fixes(rls="on_no_policy")) as (base, _):
        assert read_cards(base, None) == []
        assert read_cards(base, ALICE) == []


def test_rls_policies_own_rows_plus_public():
    with A.running(fixes(rls="on_policies")) as (base, _):
        assert read_cards(base, None) == [1]
        # 시작 데이터: alice 카드 id 1~3(공개는 1), bob 4~5. 정책을 켜면 alice는 1, 2, 3, bob은 공개 1 + 자기 4, 5.
        assert read_cards(base, ALICE) == [1, 2, 3]
        assert read_cards(base, BOB) == [1, 4, 5]


def test_with_check_blocks_writing_as_someone_else():
    with A.running(fixes(rls="on_policies")) as (base, st):
        code, _, _ = A.request(base, "/rest/v1/cards", "POST", {"owner_id": "u-bob", "title": "x"},
                               token=ALICE, apikey=A.PUBLISHABLE_KEY)
        assert code == 403
        code, _, _ = A.request(base, "/rest/v1/cards?id=2", "PATCH", {"owner_id": "u-bob"},
                               token=ALICE, apikey=A.PUBLISHABLE_KEY)
        assert code == 403
        # 남의 카드는 USING에 걸려 보이지 않으므로 오류가 아니라 updated 0. 반대로 자기 카드의 주인을 바꾸는 것은 WITH CHECK 위반이라 403.
        code, _, body = A.request(base, "/rest/v1/cards?id=4", "PATCH", {"title": "x"},
                                  token=ALICE, apikey=A.PUBLISHABLE_KEY)
        assert code == 200 and json.loads(body) == {"updated": 0}   # 보이지 않는 행은 조용히 0행
        code, _, body = A.request(base, "/rest/v1/cards?id=5", "DELETE", token=ALICE, apikey=A.PUBLISHABLE_KEY)
        assert json.loads(body) == {"deleted": 0}
        assert [r["owner_id"] for r in st.cards.all_rows()] == ["u-alice"] * 3 + ["u-bob"] * 2


def test_identity_comes_from_session_not_body():
    # (본문이 아니라 서버가 정한 uid로 판정하는지 모형 수준에서 확인한다)
    """요청 본문에 owner_id를 적어도, 판정은 서버가 토큰에서 꺼낸 uid로 한다."""
    t = L.CardsTable("on_policies")
    with pytest.raises(L.Forbidden):
        t.insert("authenticated", "u-alice", {"owner_id": "u-bob", "title": "x"})
    with pytest.raises(L.Forbidden):
        t.insert("anon", None, {"owner_id": "u-alice", "title": "x"})


def test_python_policy_matches_sqlite_where_and_sql_file():
    for mode in ("off", "on_no_policy", "on_policies"):
        t = L.CardsTable(mode)
        for role, uid in (("anon", None), ("authenticated", "u-alice"), ("authenticated", "u-bob")):
            assert t.select(role, uid) == t.select_sql(role, uid)
    assert L.policies_in_sql_file() == [(p["name"], p["cmd"]) for p in L.POLICIES]


def test_api_key_required():
    with A.running(A.AFTER) as (base, _):
        assert A.request(base, "/rest/v1/cards")[0] == 401


# ---------- 2. AI 키 ----------

def test_secret_key_only_in_before_bundle():
    for f, expected in ((A.BEFORE, 1), (A.AFTER, 0)):
        with A.running(f) as (base, _):
            blob = b"".join(A.request(base, p)[2] for p in ("/", "/app.js"))
        assert blob.count(A.AI_SECRET_KEY.encode()) == expected
        # 공개용 키는 둘 다 번들에 있어도 되는 값이다. 비밀 키만 before에서 1번 나타나야 한다.
        assert blob.count(A.PUBLISHABLE_KEY.encode()) == 1   # 공개용 키는 둘 다 있다(2장)
        assert b'"/admin"' in blob                          # 관리자 주소는 숨겨지지 않는다


# ---------- 3. 관리자 ----------

@pytest.mark.parametrize("method,path", [("GET", "/admin"), ("GET", "/admin/api/users"),
                                         ("POST", "/admin/api/cards/delete?id=5")])
def test_admin_authorization(method, path):
    with A.running(A.BEFORE) as (base, _):
        assert A.request(base, path, method)[0] == 200
    # 로그인 안 함 401(누구인지 모름), 일반 사용자 403(누구인지는 알지만 권한 없음), 관리자 200.
    expected = {None: 401, ALICE: 403, ADMIN: 200}
    for token, code in expected.items():
        with A.running(A.AFTER) as (base, _):
            assert A.request(base, path, method, token=token)[0] == code


def test_admin_ignores_role_sent_by_browser():
    with A.running(A.AFTER) as (base, _):
        assert A.request(base, "/admin/api/users", token=ALICE, headers={"X-User-Role": "admin"})[0] == 403


# ---------- 4. 업로드 ----------

def test_magic_bytes():
    assert U.sniff(U.tiny_png()) == "image/png"
    assert U.sniff(U.jpeg_head()) == "image/jpeg"
    assert U.sniff(U.webp_head()) == "image/webp"
    assert U.sniff(U.HTML_TEXT) is None
    # WebP 가운데 4바이트는 아무 값이어도 된다(마스크 00)
    # WebP의 파일 크기 자리가 아무 값이어도 통과해야 한다(마스크 00 구간 시험).
    assert U.sniff(b"RIFF\xff\xff\xff\xffWEBPVP8 ") == "image/webp"


def test_tiny_png_is_real_png():
    import struct
    import zlib
    png = U.tiny_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    ln = struct.unpack(">I", png[8:12])[0]
    # PNG 구조를 직접 확인: 8바이트 서명, IHDR의 가로x세로 = 1x1, 청크 끝 CRC-32.
    assert png[12:16] == b"IHDR" and struct.unpack(">II", png[16:24]) == (1, 1)
    assert struct.unpack(">I", png[16 + ln:20 + ln])[0] == zlib.crc32(png[12:16 + ln])


def test_upload_before_accepts_all_after_accepts_three():
    acc = {"before": [], "after": []}
    for label, name, ctype, data in U.samples():
        for mode, f in (("before", A.BEFORE), ("after", A.AFTER)):
            with A.running(f) as (base, _):
                code, _, body = A.request(base, "/api/upload?name=" + quote(name), "POST", data,
                                          token=ALICE, headers={"Content-Type": ctype})
            if json.loads(body).get("ok"):
                acc[mode].append(label)
    # 시험 파일 11개: 고치기 전은 전부 받고, 고친 뒤는 진짜 형식의 PNG, JPEG 머리, WebP 머리 3개만 받는다.
    assert len(acc["before"]) == 11
    assert acc["after"] == ["정상 PNG", "정상 JPEG 머리", "정상 WebP 머리"]


def test_upload_after_stored_outside_public_with_server_name():
    with A.running(A.AFTER) as (base, st):
        code, _, body = A.request(base, "/api/upload?name=autumn-bg.png", "POST", U.tiny_png(),
                                  token=ALICE, headers={"Content-Type": "image/png"})
        url = json.loads(body)["url"]
        assert url.startswith("/files/") and "autumn" not in url
        c2, h2, _ = A.request(base, url)
        assert c2 == 200 and h2["Content-Type"] == "image/png" and h2["X-Content-Type-Options"] == "nosniff"
        assert A.request(base, "/uploads/autumn-bg.png")[0] == 404
        assert st.public_uploads == {}


def test_upload_before_serves_html_from_our_origin():
    with A.running(A.BEFORE) as (base, _):
        A.request(base, "/api/upload?name=notes.html", "POST", U.HTML_TEXT, token=ALICE,
                  headers={"Content-Type": "text/html"})
        code, h, _ = A.request(base, "/uploads/notes.html")
        assert code == 200 and h["Content-Type"].startswith("text/html")


def test_size_limit_boundary():
    png = U.tiny_png()
    # 경계값 시험: 정확히 MAX_BYTES(2 MiB)는 통과, 1바이트 더 크면 "크기 초과"(off-by-one 확인).
    exact = png + b"\x00" * (U.MAX_BYTES - len(png))
    assert U.validate("a.png", "image/png", exact)["ok"]
    assert "크기 초과" in U.validate("a.png", "image/png", exact + b"\x00")["reasons"]


def test_upload_requires_login():
    with A.running(A.AFTER) as (base, _):
        assert A.request(base, "/api/upload?name=a.png", "POST", U.tiny_png(),
                         headers={"Content-Type": "image/png"})[0] == 401


# ---------- 5. 사용량 제한 ----------

def test_bucket_hand_calculation():
    # 0.5초 간격 30번(0~14.5초). 손계산: 처음 5번(0, 0.5, 1, 1.5, 2초) 허용, 다음 토큰은 12초에 생겨 총 6번.
    times = [R.F(k, 2) for k in range(30)]
    sim = R.simulate(R.bucket_take(), times)
    assert sim["allowed"] == 6 and sim["allowed_at"] == [0, R.F(1, 2), 1, R.F(3, 2), 2, 12]
    # 2.5초의 요청: 토큰 5/24개 남음 -> (1 - 5/24) x 12 = 9.5초 -> 올림 10.
    assert sim["first_retry_after"] == 10
    # 상한 B + rT = 5 + 14.5/12 = 149/24 = 약 6.2 -> 허용 6번 이하.
    assert R.upper_bound(R.F(29, 2)) == R.F(149, 24)


def test_bucket_one_hour():
    sim = R.simulate(R.bucket_take(), [R.F(k) for k in range(3600)])
    # 1시간 동안 1초마다: 5 + 3599/12 = 304.9 -> 304번. 단가 50원이면 15,200원(가상 단가).
    assert sim["allowed"] == 304 == math.floor(R.upper_bound(3599))


def test_bucket_never_exceeds_bound():
    # 요청 간격(gap_num/4초) x 횟수 조합을 훑어 허용 수가 항상 B + rT 이하인지 확인(성질 시험).
    for gap_num in (1, 2, 3, 5, 7, 12, 13, 30):
        for n in (1, 10, 50, 200):
            times = [R.F(gap_num, 4) * k for k in range(n)]
            sim = R.simulate(R.bucket_take(), times)
            assert sim["allowed"] <= math.floor(R.upper_bound(times[-1]))


def test_per_user_buckets_are_independent():
    clock = R.FakeClock(0)
    p = R.PerUserLimiter(clock)
    for k in range(20):
        clock.set(R.F(k, 2))
        p.take("alice")
    clock.set(10)
    assert p.take("bob")[0] is True
    # 전체 횟수 방식은 alice가 한도를 다 쓰면 bob까지 막힌다. 사용자별 버킷은 bob에 영향이 없다(bob은 10초 시점에 첫 요청).
    g = R.GlobalCounter(20)
    for _ in range(20):
        g.take("alice")
    assert g.take("bob")[0] is False


def test_http_429_and_retry_after():
    clock = R.FakeClock(0)
    codes, retry = [], None
    with A.running(A.AFTER, clock) as (base, st):
        for k in range(30):
            clock.set(R.F(k, 2))
            code, h, _ = A.request(base, "/api/generate-image", "POST", {"prompt": "가을"}, token=ALICE)
            codes.append(code)
            if code == 429 and retry is None:
                retry = h["Retry-After"]
        assert st.generated == 6
    # 6번 허용(생성 호출 6번) + 24번 거부 = 30번. 첫 429의 Retry-After는 10초(문자열 "10").
    assert codes.count(200) == 6 and codes.count(429) == 24 and retry == "10"


# ---------- 결과 파일 ----------

def test_results_json_matches_modules():
    res = json.loads((HERE / "results" / "ch03.json").read_text(encoding="utf-8"))
    assert res["e0_checklist"]["before_pass"] == 0 and res["e0_checklist"]["after_pass"] == 5
    assert res["e1_rls"]["on_policies"]["read"]["alice"]["ids"] == [1, 2, 3]
    assert res["e4_upload"]["summary"]["accepted_after"] == 3
    assert res["e5_rate_limit"]["s1_burst_30_in_15s"]["allowed_bucket"] == 6
    assert res["e5_rate_limit"]["s2_1_per_sec_for_1h"]["cost_won_bucket"] == 304 * 50
    assert res["e6_crosscheck"]["rls_python_vs_sqlite"]["mismatches"] == 0
    # 포트 번호는 실행마다 달라지므로 결과 파일에 남으면 재현이 안 된다.
    assert "127.0.0.1:" not in json.dumps(res)   # 포트 번호처럼 매번 바뀌는 값은 남기지 않는다
