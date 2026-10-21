import pytest

from deployment_choice import evaluate, simultaneous_completion_times, traffic_summary


def base_profile(**updates):
    profile = {
        "region": "ap-northeast-2",
        "requests_per_day": 20,
        "days_per_month": 30,
        "seconds_per_request": 30,
        "startup_seconds": 90,
        "acceptable_wait_seconds": 300,
        "peak_concurrency": 20,
        "runtime_memory_gb": 8,
        "container_image_gb": 6,
        "payload_mb": 0.02,
        "gpu_required": True,
        "async_allowed": True,
    }
    profile.update(updates)
    return profile


def test_traffic_arithmetic_matches_chapter():
    summary = traffic_summary(base_profile())
    assert summary["monthly_requests"] == 600
    assert summary["active_hours"] == pytest.approx(5)
    assert summary["always_on_hours"] == pytest.approx(720)
    assert summary["always_on_to_active_ratio"] == pytest.approx(144)


def test_twenty_simultaneous_requests_show_queue_and_startup():
    warm = simultaneous_completion_times(20, 30)
    cold = simultaneous_completion_times(20, 30, startup_seconds=90)
    assert [warm[0], warm[9], warm[19]] == [30, 300, 600]
    assert [cold[0], cold[9], cold[19]] == [120, 390, 690]


def test_gpu_workload_blocks_cpu_only_options():
    result = evaluate(base_profile())
    assert "lambda_default_compute" in result["blocked"]
    assert "ecs_fargate" in result["blocked"]
    assert "sagemaker_serverless" in result["blocked"]
    assert "sagemaker_async" in result["candidates"]
    assert "sagemaker_realtime" in result["candidates"]


def test_small_cpu_workload_keeps_serverless_candidates():
    result = evaluate(base_profile(
        gpu_required=False,
        runtime_memory_gb=2,
        container_image_gb=1,
        seconds_per_request=5,
        startup_seconds=2,
        acceptable_wait_seconds=10,
    ))
    assert "lambda_default_compute" in result["candidates"]
    assert "sagemaker_serverless" in result["candidates"]


def test_missing_measurements_hold_the_decision():
    profile = base_profile(runtime_memory_gb=None, gpu_required=None, region=None)
    result = evaluate(profile)
    assert result["decision_status"] == "HOLD"
    assert set(result["missing_measurements"]) == {"runtime_memory_gb", "gpu_required", "region"}


def test_bad_queue_inputs_fail():
    with pytest.raises(ValueError):
        simultaneous_completion_times(0, 30)
