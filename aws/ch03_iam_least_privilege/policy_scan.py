"""IAM 정책 JSON을 읽어 과한 권한을 찾는다. AWS 호출 없음, 표준 라이브러리만 쓴다.

판정은 학습용 규칙이다. 실제 정책 검증은 IAM Access Analyzer 같은 공식 도구로 다시 한다.
"""

from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def as_list(x) -> list:
    return x if isinstance(x, list) else [x]


def statements(policy: dict) -> list[dict]:
    return as_list(policy.get("Statement", []))


def scan(policy: dict) -> list[dict]:
    """Allow 문장마다 위험 신호를 찾는다. level: HIGH(바로 고칠 것), REVIEW(이유가 있는지 확인)."""
    found = []
    for i, st in enumerate(statements(policy)):
        if st.get("Effect") != "Allow":
            continue
        actions = [a.lower() for a in as_list(st.get("Action", []))]
        resources = as_list(st.get("Resource", []))
        for a in actions:
            if a == "*":
                found.append({"stmt": i, "level": "HIGH", "why": "모든 서비스의 모든 동작(Action \"*\")"})
            elif a.endswith(":*"):
                found.append({"stmt": i, "level": "REVIEW", "why": f"{a.split(':')[0]} 서비스의 모든 동작({a})"})
            if a == "iam:passrole" and "*" in resources:
                found.append({"stmt": i, "level": "HIGH", "why": "어떤 역할이든 넘길 수 있는 iam:PassRole"})
        if "*" in resources and "*" not in actions:
            found.append({"stmt": i, "level": "REVIEW", "why": "모든 리소스(Resource \"*\"). 서비스가 리소스 단위 권한을 지원하지 않을 수도 있어요"})
        if "*" in resources and "*" in actions:
            found.append({"stmt": i, "level": "HIGH", "why": "모든 동작을 모든 리소스에(Action \"*\" + Resource \"*\")"})
        if st.get("Condition") is None and ("*" in actions or "*" in resources):
            found.append({"stmt": i, "level": "REVIEW", "why": "넓은 범위인데 Condition이 없어요"})
    return found


def arn_match(pattern: str, arn: str) -> bool:
    """* 는 /, : 를 넘어 이어질 수 있다(공식 문서 예: bucket/*/test/*). fnmatch의 * 와 같다."""
    return fnmatch.fnmatchcase(arn, pattern)


def allowed(policy: dict, action: str, resource: str) -> str:
    """기본 거부. 명시적 Deny가 Allow를 이긴다. 반환: ALLOW / DENY(명시) / DENY(기본)."""
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
    scoped = data["scoped"]
    for act, res in [("s3:GetObject", "arn:aws:s3:::example-model-bucket/reply-draft/v1/model.gguf"),
                     ("s3:GetObject", "arn:aws:s3:::example-model-bucket/other/secret.txt"),
                     ("s3:DeleteObject", "arn:aws:s3:::example-model-bucket/reply-draft/v1/model.gguf")]:
        print(f"scoped {act} {res.split(':::')[1]} -> {allowed(scoped, act, res)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "policies.json"))
