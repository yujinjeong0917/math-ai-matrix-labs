"""프롬프트 로그의 마스킹과 보관 기간을 AWS 없이 점검한다.

마스킹은 정규식 몇 개로 만든 학습용 모형이다. 이름·주소 같은 자유 문장은 잡지 못한다.
보관 기간 값은 CloudWatch Logs 공식 문서(PutRetentionPolicy)에서 확인한 목록이다.

재는 것: (1) 개인정보 형식 마스킹, (2) 보관 기간별 저장량과 삭제 지연을 더한 최대 잔존 일수,
(3) 마스킹 정책을 늦게 켰을 때 마스킹되지 않은 건수, (4) 로그 그룹 설정의 위반.
입력: log_config.json(하루 2만 건, 건당 2만 바이트 등 가상 값). 출력: README 기대 출력
(7일 2.8GB와 90일 36.0GB, 10% 표본 0.28GB, 60000건, debug 그룹의 만료 없음 지적 2줄).
모든 번호와 이메일은 가짜 값이며 법령상 보관 기준은 다루지 않는다.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

GB = 1_000_000_000

# CloudWatch Logs 로그 그룹 보관 기간으로 쓸 수 있는 값(일). 확인일 2026-11-14.
ALLOWED_RETENTION_DAYS = (
    1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545,
    731, 1096, 1827, 2192, 2557, 2922, 3288, 3653,
)
ACCOUNT_POLICY_MIN_DAYS = 30
# 설정 보관 기간 + 삭제 지연이 "최대 잔존 일수"다(7일 설정이면 최대 10일)
DELETE_LAG_DAYS = 3  # 보관 기간이 지난 뒤 실제 삭제까지 보통 최대 72시간

# 형식만 보는 정규식이다. 번호가 실제로 유효한지는 검증하지 않는다.
# 앞에서부터 차례로 치환하므로 순서가 결과에 영향을 줄 수 있다(겹치는 형식이 있으면 먼저 오는 쪽이 이김)
PATTERNS = (
    ("주민번호형", re.compile(r"\b\d{6}-[1-4]\d{6}\b")),
    ("카드번호형", re.compile(r"\b(?:\d{4}[- ]){3}\d{4}\b")),
    ("전화", re.compile(r"\b01[016789]-?\d{3,4}-?\d{4}\b")),
    ("이메일", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
)


def mask_text(text: str) -> tuple[str, dict[str, int]]:
    """패턴별로 [라벨]로 바꾼 문장과 라벨별 치환 횟수를 돌려준다. 이름·주소 같은 자유 문장은 못 잡는다."""
    counts: dict[str, int] = {}
    for label, pattern in PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = n
    return text, counts


def worst_case_age_days(retention_days: int) -> int:
    """설정 보관 기간에 삭제 지연(3일)을 더한 최대 잔존 일수."""
    return retention_days + DELETE_LAG_DAYS


def stored_gb(requests_per_day: float, bytes_per_request: float, retention_days: float, sample_percent: float = 100) -> float:
    """정상 상태 저장량(GB, 10^9 바이트): 하루 바이트 x 표본 비율 x 보관 일수. 2만 건 x 2만 바이트 x 7일 = 2.8GB."""
    return requests_per_day * bytes_per_request * sample_percent / 100 * retention_days / GB


def unmasked_events(requests_per_day: float, protection_policy_days_ago: float) -> float:
    """마스킹 정책은 쌓일 때 적용되므로, 정책을 켜기 전에 들어온 로그 건수는 그대로 남는다."""
    return requests_per_day * protection_policy_days_ago


def check_log_groups(config: dict[str, Any]) -> list[str]:
    """로그 그룹 설정의 위반 문구 목록. 빈 목록이면 통과."""
    findings: list[str] = []
    account = config.get("account_policy_days")
    if account is not None and account < ACCOUNT_POLICY_MIN_DAYS:
        findings.append(f"계정 단위 보관 정책은 최소 {ACCOUNT_POLICY_MIN_DAYS}일이에요: {account}일")
    for g in config["log_groups"]:
        days = g["retention_days"]
        name = g["name"]
        if days is None:
            findings.append(f"{name}: 만료 없음(저장이 계속 늘어요)")
            effective = None
        elif days not in ALLOWED_RETENTION_DAYS:
            findings.append(f"{name}: {days}일은 허용된 보관 기간이 아니에요")
            effective = days
        else:
            effective = days
        # 계정 정책이 있으면 그룹 설정을 덮어쓴다고 가정하고 실제 보관 기간(effective)을 계정 값으로 본다
        # (덮어쓰기 규칙은 이 실습의 모형, 실제 동작은 공식 문서로 확인 필요)
        if account is not None and account >= ACCOUNT_POLICY_MIN_DAYS:
            if effective != account:
                findings.append(f"{name}: 계정 정책 {account}일이 그룹 설정을 덮어써요")
            effective = account
        # 원문 프롬프트가 든 그룹만 검사. 만료 없음(None)은 무한대와 같아 무조건 위반
        if g["contains_raw_prompt"] and (effective is None or effective > config["max_raw_prompt_days"]):
            shown = "만료 없음" if effective is None else f"{effective}일"
            findings.append(f"{name}: 원문 프롬프트가 허용 {config['max_raw_prompt_days']}일보다 오래 남아요({shown})")
    return findings


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(__file__).with_name("log_config.json")
    c = json.loads(path.read_text(encoding="utf-8"))
    sample = "문의: 홍길동(010-1234-5678, test@example.com) 카드 1234-5678-9012-3456 환불해 주세요"
    masked, counts = mask_text(sample)
    print("마스킹 전:", sample)
    print("마스킹 후:", masked, counts)
    for days in (7, 90):
        print(f"원문 로그 보관 {days}일: 최대 {worst_case_age_days(days)}일까지 남을 수 있음, 저장 {stored_gb(c['requests_per_day'], c['raw_prompt_bytes_per_request'], days):.1f}GB")
    print(f"10%만 남기면 7일 저장 {stored_gb(c['requests_per_day'], c['raw_prompt_bytes_per_request'], 7, c['sample_percent']):.2f}GB")
    print(f"정책 {c['protection_policy_days_ago']}일 전에 켰다면 마스킹 안 된 건수 {unmasked_events(c['requests_per_day'], c['protection_policy_days_ago']):.0f}건")
    print(f"점검 기준: 원문 그룹의 설정 보관 기간이 {c['max_raw_prompt_days']}일 이하여야 통과 (삭제 지연까지 더하면 실제로는 최대 {worst_case_age_days(c['max_raw_prompt_days'])}일 남을 수 있음)")
    for f in check_log_groups(c):
        print("점검:", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
