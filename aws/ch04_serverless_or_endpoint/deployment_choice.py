"""AWS 배포 후보를 가격이 아닌 실행 제약으로 먼저 거른다.

이 규칙은 AWS 공식 문서를 2026-10-22에 확인한 학습용 스냅샷이다.
실제 배포 전에는 리전·계정 할당량·최신 문서를 다시 확인한다.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


REQUIRED_MEASUREMENTS = ("runtime_memory_gb", "gpu_required", "container_image_gb", "region")


def traffic_summary(profile: dict[str, Any]) -> dict[str, float]:
    monthly_requests = profile["requests_per_day"] * profile["days_per_month"]
    active_seconds = monthly_requests * profile["seconds_per_request"]
    always_on_seconds = profile["days_per_month"] * 24 * 60 * 60
    return {
        "monthly_requests": monthly_requests,
        "active_seconds": active_seconds,
        "active_hours": active_seconds / 3600,
        "always_on_hours": always_on_seconds / 3600,
        "always_on_to_active_ratio": always_on_seconds / active_seconds,
    }


def simultaneous_completion_times(
    requests: int, seconds_per_request: float, startup_seconds: float = 0, workers: int = 1
) -> list[float]:
    """모든 요청이 동시에 왔을 때 각 요청의 완료 시간을 구한다."""
    if requests < 1 or seconds_per_request <= 0 or startup_seconds < 0 or workers < 1:
        raise ValueError("요청 수·건당 시간·시작 시간·작업자 수를 확인하세요")
    return [startup_seconds + math.ceil(i / workers) * seconds_per_request for i in range(1, requests + 1)]


def _add(result: dict[str, Any], name: str, reasons: list[str], note: str) -> None:
    bucket = "blocked" if reasons else "candidates"
    result[bucket][name] = reasons if reasons else [note]


def evaluate(profile: dict[str, Any]) -> dict[str, Any]:
    """실행 가능성과 대기 방식으로 후보를 거른다. 최종 구매 결정은 아니다."""
    missing = [key for key in REQUIRED_MEASUREMENTS if profile.get(key) is None]
    result: dict[str, Any] = {
        "missing_measurements": missing,
        "traffic": traffic_summary(profile),
        "candidates": {},
        "blocked": {},
        "decision_status": "HOLD" if missing else "READY_FOR_BENCHMARK",
    }
    if missing:
        return result

    gpu = profile["gpu_required"]
    memory = profile["runtime_memory_gb"]
    image = profile["container_image_gb"]
    duration = profile["seconds_per_request"]
    payload = profile["payload_mb"]
    async_allowed = profile["async_allowed"]
    wait = profile["acceptable_wait_seconds"]
    startup = profile["startup_seconds"]

    lambda_reasons = []
    if gpu:
        lambda_reasons.append("GPU 선택지가 없음")
    if memory > 10:
        lambda_reasons.append("실행 메모리가 Lambda 10GiB 상한을 넘음")
    if image > 10:
        lambda_reasons.append("컨테이너 이미지가 10GB 상한을 넘음")
    if duration > 900:
        lambda_reasons.append("한 번 실행이 15분을 넘음")
    if startup > wait:
        lambda_reasons.append("시작 시간이 허용 대기 시간보다 길 수 있음")
    _add(result, "lambda_default_compute", lambda_reasons, "짧은 CPU 작업으로 실측 벤치마크")

    fargate_reasons = ["Fargate 작업은 GPU 매개변수를 지원하지 않음"] if gpu else []
    _add(result, "ecs_fargate", fargate_reasons, "긴 CPU 컨테이너와 지속 연결 필요를 벤치마크")

    serverless_reasons = []
    if gpu:
        serverless_reasons.append("SageMaker Serverless는 GPU를 지원하지 않음")
    if memory > 6:
        serverless_reasons.append("실행 메모리가 6GB 상한을 넘음")
    if image > 10:
        serverless_reasons.append("컨테이너 이미지가 10GB 상한을 넘음")
    if duration >= 60:
        serverless_reasons.append("요청 처리가 60초 미만 조건에 들지 않음")
    if payload > 4:
        serverless_reasons.append("요청 페이로드가 4MB 상한을 넘음")
    if startup > wait:
        serverless_reasons.append("콜드 스타트를 포함한 시작 시간이 허용 대기를 넘을 수 있음")
    _add(result, "sagemaker_serverless", serverless_reasons, "간헐적 CPU 추론으로 실측 벤치마크")

    async_reasons = []
    if not async_allowed:
        async_reasons.append("사용자가 나중에 답을 받는 흐름을 허용하지 않음")
    if duration > 3600:
        async_reasons.append("한 번 처리가 1시간을 넘음")
    if payload > 1024:
        async_reasons.append("입력이 1GB를 넘음")
    _add(result, "sagemaker_async", async_reasons, "GPU 가능·0까지 축소, 첫 요청 대기를 실측")

    _add(result, "sagemaker_realtime", [], "낮고 일정한 응답 시간과 유휴 비용을 함께 벤치마크")
    return result


def main() -> None:
    profile_path = Path(sys.argv[1] if len(sys.argv) > 1 else "traffic_profile.json")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(profile), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
