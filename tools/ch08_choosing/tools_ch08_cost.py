"""1~7장 결과를 한 달 비용 표로 모은다.

한 달 = 한 주 x 52/12 (1년 52주를 12달로 나눔, 1장과 같은 규칙).
행마다 무엇을 셌는지와 단위가 다르다. 행끼리 더하거나 순위를 매기지 않는다.
API 사용료만 센다. 도구 요금제(좌석)·구독료는 넣지 않는다.
"""

from fractions import Fraction as F

import tools_ch08_data as D

# 식: 한 달 = 52주 / 12달 = 약 4.33주. 달력의 월 길이 차이는 무시한 단순화.
WEEKS_PER_MONTH = F(52, 12)
# 4장 지시문 토큰을 달러로 바꿀 때 쓰는 값: Claude Sonnet 5.5 기본 입력가(1장·2장 가격표, 100만 토큰당 2달러).
SONNET_INPUT_PER_MTOK = F(2)


# 값을 str로 거쳐 Fraction으로 바꾼다. float를 바로 넣으면 0.8556 같은 값이 이진 근사값 그대로 분수가 되어 곱셈 뒤 반올림이 달라질 수 있다.
def month(week):
    return F(str(week)) * WEEKS_PER_MONTH


# 표시용: 소수 넷째 자리 반올림. 계산 중에는 쓰지 않고 결과를 담을 때만 쓴다.
def r4(x):
    return round(float(x), 4)


def table(up=None):
    up = up or D.load_upstream()
    rows = []

    def usd(ch, setup, counted, week):
        rows.append({"chapter": ch, "setup": setup, "counted": counted, "unit": "USD",
                     "week": float(F(str(week))), "month": r4(month(week))})

    usd(1, "문의 묶음을 매번 붙여 넣어 12번 호출, 캐싱 없음 (Claude Sonnet 5.5)",
        "API 입력+출력", up["ch01_week_paste"])
    usd(2, "같은 12번 호출에 프롬프트 캐싱 (5분 보관)",
        "API 입력+출력", up["ch02_week_cache"])
    usd(3, "도구 호출로 문의를 읽음, 25번 호출, 캐싱 없음",
        "API 입력+출력", up["ch03_week_tool_nocache"])
    usd(3, "도구 호출 + 캐싱",
        "API 입력+출력", up["ch03_week_tool_cache"])
    # 지시문 토큰만 달러로 환산(캐싱 없음 가정). 4장은 원래 토큰을 센 장이라 이 값은 8장이 정한 환산이다.
    paste = F(up["ch04_week_instr_paste_tokens"]) * SONNET_INPUT_PER_MTOK / 1_000_000
    staged = F(up["ch04_week_instr_staged_tokens"]) * SONNET_INPUT_PER_MTOK / 1_000_000
    rows.append({"chapter": 4, "setup": "대화 15번에 지시문 전부 붙이기 (지시문 몫만)",
                 "counted": "지시문 입력 토큰을 8장에서 2달러/100만 토큰으로 바꾼 값(캐싱 없음)", "unit": "USD",
                 "tokens_week": up["ch04_week_instr_paste_tokens"], "week": r4(paste), "month": r4(paste * WEEKS_PER_MONTH)})
    rows.append({"chapter": 4, "setup": "스킬로 필요할 때만 읽기 (지시문 몫만)",
                 "counted": "위와 같음", "unit": "USD",
                 "tokens_week": up["ch04_week_instr_staged_tokens"], "week": r4(staged), "month": r4(staged * WEEKS_PER_MONTH)})
    # 5장 행의 단위는 달러가 아니라 호출 수. 한 달 호출 수를 좌석 한 달 한도와 비교하는 용도라 위의 달러 행과 더하지 않는다.
    rows.append({"chapter": 5, "setup": "Notion·Figma MCP 왕복 매주 1번",
                 "counted": "Figma 도구 호출 수(View·Collab 좌석 한 달 한도와 비교)", "unit": "calls",
                 "week": up["ch05_figma_calls_per_run"], "month": up["ch05_figma_calls_month"],
                 "limit": up["ch05_figma_limit_view_collab"]})
    # 6장 행: 'week'는 사람 승인 횟수, 'month'는 그 52/12배. 확률 열은 끝까지 옳은 비율. 달러가 아니다.
    rows.append({"chapter": 6, "setup": "에이전트에게 22단계 반복 맡기기 (단계 확인 + 다시 하기 + 알림 앞 사람 승인)",
                 "counted": "끝까지 옳은 주의 비율(확인 없음 → 확인 있음)과 사람 승인 횟수, 달러 아님", "unit": "prob/approvals",
                 "correct_none": up["ch06_correct_none"], "correct_checked": r4(up["ch06_correct_checked"]),
                 "wrong_checked": r4(up["ch06_wrong_checked"]), "attempts_week": r4(up["ch06_attempts_checked"]),
                 "week": up["ch06_approvals_week"], "month": r4(month(up["ch06_approvals_week"]))})
    # 7장 행은 값이 범위([최소, 최대]): 모델마다 한 번 뽑기 값이 달라 제한 없이 뽑을 때의 기대 비용 중 가장 싼 모델과 가장 비싼 모델을 적는다. 주 1편은 가정.
    rows.append({"chapter": 7, "setup": "15초 영상 한 편을 매주 (마음에 들 확률 " + up["ch07_p"] + " 가정, 될 때까지 뽑기)",
                 "counted": "영상 생성 사용료 기댓값, 모델에 따라 범위", "unit": "USD",
                 "week": [up["ch07_video_min"], up["ch07_video_max"]],
                 "month": [r4(month(up["ch07_video_min"])), r4(month(up["ch07_video_max"]))],
                 "weekly_is_assumption": True})
    rows.append({"chapter": 7, "setup": "표지 그림 한 장을 매주 (같은 가정)",
                 "counted": "이미지 생성 사용료 기댓값, 모델에 따라 범위", "unit": "USD",
                 "week": [up["ch07_image_min"], up["ch07_image_max"]],
                 "month": [r4(month(up["ch07_image_min"])), r4(month(up["ch07_image_max"]))],
                 "weekly_is_assumption": True})
    return rows


def hand():
    """첫 화면 손계산: 0.8556달러 x 52 / 12."""
    up = D.load_upstream()
    # 손계산: 0.8556 x 52 = 44.4912, / 12 = 3.7076달러(1장과 같은 규칙).
    w = F(str(up["ch01_week_paste"]))
    return {"week": float(w), "times52": r4(w * 52), "month": r4(w * 52 / 12),
            # 모델만 바꿨을 때의 폭: 가장 싼 모델과 가장 비싼 모델의 한 달 비용 비율.
            "model_range": {"haiku": up["ch01_month_haiku"], "astra": up["ch01_month_astra"],
                            "ratio": round(up["ch01_month_astra"] / up["ch01_month_haiku"], 1)}}
