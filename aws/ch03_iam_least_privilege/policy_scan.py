"""IAM 정책 JSON을 읽어 과한 권한을 찾는다. AWS 호출 없음, 표준 라이브러리만 쓴다.

판정은 학습용 규칙이다. 실제 정책 검증은 IAM Access Analyzer 같은 공식 도구로 다시 한다.

재는 것: (1) Allow 문장의 과한 권한 신호(scan)와 점수, (2) 정책 하나 안에서 한 동작이 허용되는지(allowed).
입력: policies.json (정책 이름 -> IAM 정책 JSON). 출력: 정책별 HIGH/REVIEW 개수와 점수,
scoped 정책의 세 가지 허용 판정 (README 기대 출력과 대응: too_broad 점수 7, half_broad 점수 8, scoped 0).
Condition, 권한 경계, NotAction 등은 계산하지 않는다.
"""

from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def as_list(x) -> list:
    """IAM은 Action·Resource·Statement를 문자열 하나 또는 목록으로 쓸 수 있어 목록으로 통일한다."""
    return x if isinstance(x, list) else [x]


def statements(policy: dict) -> list[dict]:
    return as_list(policy.get("Statement", []))


def scan(policy: dict) -> list[dict]:
    """Allow 문장마다 위험 신호를 찾는다. level: HIGH(바로 고칠 것), REVIEW(이유가 있는지 확인)."""
    found = []
    for i, st in enumerate(statements(policy)):
        # Deny 문장은 권한을 줄이는 쪽이라 위험 신호로 세지 않는다(테스트 참고)
        if st.get("Effect") != "Allow":
            continue
        # 동작 이름은 대소문자를 구분하지 않으므로 소문자로 맞춰 비교한다
        actions = [a.lower() for a in as_list(st.get("Action", []))]
        resources = as_list(st.get("Resource", []))
        for a in actions:
            if a == "*":
                found.append({"stmt": i, "level": "HIGH", "why": "모든 서비스의 모든 동작(Action \"*\")"})
            elif a.endswith(":*"):
                found.append({"stmt": i, "level": "REVIEW", "why": f"{a.split(':')[0]} 서비스의 모든 동작({a})"})
            # PassRole은 서비스에 역할을 넘기는 권한이라 리소스가 *이면 권한 상승 경로가 될 수 있다
            if a == "iam:passrole" and "*" in resources:
                found.append({"stmt": i, "level": "HIGH", "why": "어떤 역할이든 넘길 수 있는 iam:PassRole"})
        # 아래 두 분기는 서로 배타적: 리소스 *만 넓으면 REVIEW, 동작도 *이면 HIGH
        if "*" in resources and "*" not in actions:
            found.append({"stmt": i, "level": "REVIEW", "why": "모든 리소스(Resource \"*\"). 서비스가 리소스 단위 권한을 지원하지 않을 수도 있어요"})
        if "*" in resources and "*" in actions:
            found.append({"stmt": i, "level": "HIGH", "why": "모든 동작을 모든 리소스에(Action \"*\" + Resource \"*\")"})
        # 조건(IP, 태그 등)으로 범위를 좁히지 않은 넓은 문장
        if st.get("Condition") is None and ("*" in actions or "*" in resources):
            found.append({"stmt": i, "level": "REVIEW", "why": "넓은 범위인데 Condition이 없어요"})
    return found


def arn_match(pattern: str, arn: str) -> bool:
    """* 는 /, : 를 넘어 이어질 수 있다(공식 문서 예: bucket/*/test/*). fnmatch의 * 와 같다."""
    # 함정: fnmatch는 [abc] 같은 문자 집합도 해석한다. IAM이 같은 규칙인지는 확인 필요(이 실습 ARN엔 쓰이지 않음)
    return fnmatch.fnmatchcase(arn, pattern)


def allowed(policy: dict, action: str, resource: str) -> str:
    """기본 거부. 명시적 Deny가 Allow를 이긴다. 반환: ALLOW / DENY(명시) / DENY(기본)."""
    # 평가 순서: 하나라도 명시적 Deny가 맞으면 즉시 거부, 아니면 Allow가 하나라도 있어야 허용, 둘 다 없으면 기본 거부
    allow = False
    for st in statements(policy):
        acts = [a.lower() for a in as_list(st.get("Action", []))]
        ress = as_list(st.get("Resource", []))
        if any(fnmatch.fnmatchcase(action.lower(), a) for a in acts) and any(arn_match(r, resource) for r in ress):
            if st.get("Effect") == "Deny":
                return "DENY(명시)"
            if st.get("Effect") == "Allow":
                allow = True
    return "ALLOW" if allow else "DENY(기본)"


def risk_score(findings: list[dict]) -> int:
    """HIGH 3점, REVIEW 1점의 합(가중치는 이 실습의 선택). 문장을 센 값이라 정책 간 순위용이 아니다."""
    return sum(3 if f["level"] == "HIGH" else 1 for f in findings)


def main(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, pol in data.items():
        if name == "note":
            continue
        f = scan(pol)
        print(f"[{name}] HIGH {sum(x['level']=='HIGH' for x in f)}, REVIEW {sum(x['level']=='REVIEW' for x in f)}, 점수 {risk_score(f)}")
        for x in f:
            print(f"   문장 {x['stmt']} {x['level']}: {x['why']}")
    # 세 질의: 허용된 경로(ALLOW), 허용되지 않은 다른 경로(DENY 기본), 정책에 Deny가 적힌 동작(DENY 명시)
    scoped = data["scoped"]
    for act, res in [("s3:GetObject", "arn:aws:s3:::example-model-bucket/reply-draft/v1/model.gguf"),
                     ("s3:GetObject", "arn:aws:s3:::example-model-bucket/other/secret.txt"),
                     ("s3:DeleteObject", "arn:aws:s3:::example-model-bucket/reply-draft/v1/model.gguf")]:
        print(f"scoped {act} {res.split(':::')[1]} -> {allowed(scoped, act, res)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "policies.json"))
