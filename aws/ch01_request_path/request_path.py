"""요청 한 건의 경로를 단계 표로 만들고, 가장 작은 시간 한도·크기 한도에 걸리는지 검사한다.

AWS 호출 없음. 한도 값은 limits_2026-11-14.json(공식 문서 확인값)에서 읽고,
단계별 시간(ms)은 모두 가상(예시용) 값이다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MIB = 1024 * 1024
HERE = Path(__file__).parent

# (이름, 시간 ms 가상, 범위). scope=outside: API 입구의 통합 제한 시간이 재지 않는 앞 구간(DNS·TLS·입구 처리),
# inside: API 입구가 뒤쪽 서비스를 기다리는 구간(통합 제한 시간이 재는 곳).
COMMON_STEPS = [
    ("DNS 조회", 50, "outside"),
    ("TLS 연결", 100, "outside"),
    ("API 입구 처리", 50, "outside"),
]
BACKEND_OVERHEAD = [("Lambda 시작(가상)", 200), ("로그 기록", 50)]  # 합 250ms = 0.25초

# 경로 이름 -> (시간 한도 키 목록, 크기 한도 키 목록)
PATHS = {
    "REST API + Lambda": (["apigw_rest_integration_timeout_s"], ["apigw_payload_mb", "lambda_sync_payload_mb"]),
    "HTTP API + Lambda": (["apigw_http_integration_timeout_s"], ["apigw_payload_mb", "lambda_sync_payload_mb"]),
    "REST API + Lambda + SageMaker 실시간": (
        ["apigw_rest_integration_timeout_s", "sagemaker_invoke_timeout_s"],
        ["apigw_payload_mb", "lambda_sync_payload_mb", "sagemaker_invoke_body_bytes"],
    ),
}


def load_limits(path: Path | None = None) -> dict[str, float]:
    data = json.loads((path or HERE / "limits_2026-11-14.json").read_text(encoding="utf-8"))
    return {k: v["value"] for k, v in data["limits"].items()}


def to_bytes(key: str, value: float) -> int:
    return int(value) if key.endswith("_bytes") else int(value * MIB)


def step_table(model_ms: int) -> list[dict]:
    rows = [{"step": n, "ms": ms, "scope": s} for n, ms, s in COMMON_STEPS]
    rows.append({"step": "모델 계산", "ms": model_ms, "scope": "inside"})
    rows += [{"step": n, "ms": ms, "scope": "inside"} for n, ms in BACKEND_OVERHEAD]
    return rows


def check(path_name: str, model_seconds: float, payload_mb: float, limits: dict[str, float],
          client_timeout_s: float | None = None) -> dict:
    time_keys, size_keys = PATHS[path_name]
    model_ms = round(model_seconds * 1000)
    rows = step_table(model_ms)
    backend_s = sum(r["ms"] for r in rows if r["scope"] == "inside") / 1000
    total_s = sum(r["ms"] for r in rows) / 1000
    tightest_key = min(time_keys, key=lambda k: limits[k])
    tightest_s = limits[tightest_key]
    payload_bytes = round(payload_mb * MIB)
    size_key = min(size_keys, key=lambda k: to_bytes(k, limits[k]))
    size_limit = to_bytes(size_key, limits[size_key])
    problems = []
    if backend_s > tightest_s:
        problems.append(f"API 입구 뒤 구간 {backend_s:.2f}초 > {tightest_key} {tightest_s}초")
    if payload_bytes > size_limit:
        problems.append(f"입력 {payload_bytes}바이트 > {size_key} {size_limit}바이트")
    if client_timeout_s is not None and total_s > client_timeout_s:
        problems.append(f"전체 {total_s:.2f}초 > 클라이언트 대기 {client_timeout_s}초")
    return {
        "path": path_name, "backend_seconds": backend_s, "total_seconds": total_s,
        "tightest_time_key": tightest_key, "tightest_time_s": tightest_s,
        "max_model_seconds": tightest_s - sum(ms for _, ms in BACKEND_OVERHEAD) / 1000,
        "size_key": size_key, "size_limit_bytes": size_limit,
        "ok": not problems, "problems": problems,
    }


def main() -> None:
    limits = load_limits()
    for r in step_table(30000):
        print(f"{r['step']:<16}{r['ms']:>7} ms  ({r['scope']})")
    print()
    for name in PATHS:
        r = check(name, 30, 0.02, limits)
        print(f"[30초 모델] {name}: {'통과' if r['ok'] else '탈락'} / 모델 최대 {r['max_model_seconds']:.2f}초 / {r['problems']}")
    for sec in (20, 28.75, 28.8):
        r = check("REST API + Lambda", sec, 0.02, limits)
        print(f"[REST {sec}초] {'통과' if r['ok'] else '탈락'} {r['problems']}")
    r = check("HTTP API + Lambda", 20, 7, limits)
    print(f"[7MiB 입력] {'통과' if r['ok'] else '탈락'} {r['problems']}")


if __name__ == "__main__":
    main()
