"""재시도와 중복 배달이 같은 작업을 몇 번 실행하게 만드는지 센다.

AWS 호출은 없다. 아래 규칙은 AWS 공식 문서에서 확인한 동작을 단순화한 모형이다.
- SQS 표준 큐는 최소 한 번 배달이라 같은 메시지가 두 번 올 수 있다.
- 메시지를 받은 순간부터 가시성 제한 시간이 흐르고, 그 안에 지우지 못하면 다시 보인다.
- Lambda 비동기 호출은 오류가 나면 기본으로 두 번 더 실행한다.
- SQS FIFO 큐는 5분 중복 제거 구간 안의 같은 중복 제거 ID를 한 번만 받는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def batch_duplicates(batch_size: int, seconds_each: float, visibility_seconds: float) -> dict[str, Any]:
    """한 작업자가 batch_size건을 한꺼번에 받아 차례로 처리할 때, 제한 시간 안에 못 지워 다시 보이게 되는 메시지."""
    if batch_size < 1 or seconds_each <= 0 or visibility_seconds <= 0:
        raise ValueError("묶음 크기·처리 시간·제한 시간을 확인하세요")
    done_at = [seconds_each * k for k in range(1, batch_size + 1)]
    duplicated = [k for k, t in enumerate(done_at, start=1) if t > visibility_seconds]
    return {
        "done_at": done_at,
        "duplicated_positions": duplicated,
        "duplicate_count": len(duplicated),
        "wasted_seconds": len(duplicated) * seconds_each,
    }


def min_visibility_seconds(batch_size: int, p95_seconds: float) -> float:
    """묶음의 마지막 메시지까지 지울 시간을 주려면 필요한 제한 시간(느린 상위 5% 건 기준)."""
    return batch_size * p95_seconds


def lambda_async_attempts(retries: int) -> int:
    return 1 + retries


def apply_effects(delivered_ids: list[str], idempotent: bool) -> int:
    """배달된 id마다 한 번 일어나는 부작용(예: 답변 발송)의 총 횟수. 멱등이면 같은 id는 한 번만."""
    if idempotent:
        return len(set(delivered_ids))
    return len(delivered_ids)


def fifo_accepted(sends: list[tuple[float, str]], window_seconds: float) -> list[tuple[float, str]]:
    """(보낸 시각, 중복 제거 ID) 목록 중 큐가 받아 주는 것. 첫 수락 시각부터 window 안의 같은 ID는 버린다."""
    first_accepted: dict[str, float] = {}
    accepted: list[tuple[float, str]] = []
    for t, dedup_id in sorted(sends):
        start = first_accepted.get(dedup_id)
        if start is None or t - start >= window_seconds:
            first_accepted[dedup_id] = t
            accepted.append((t, dedup_id))
    return accepted


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(__file__).with_name("retry_profile.json")
    p = json.loads(path.read_text(encoding="utf-8"))
    b = batch_duplicates(p["batch_size"], p["seconds_per_request"], p["visibility_timeout_seconds"])
    print(f"묶음 {p['batch_size']}건, 건당 {p['seconds_per_request']}초, 제한 시간 {p['visibility_timeout_seconds']}초")
    print(f"  완료 시각: {[round(t) for t in b['done_at']]}")
    print(f"  다시 보이는 메시지 {b['duplicate_count']}건 (순번 {b['duplicated_positions']}), 낭비 계산 {b['wasted_seconds']:.0f}초")
    need = min_visibility_seconds(p["batch_size"], p["p95_seconds_per_request"])
    print(f"  느린 건({p['p95_seconds_per_request']}초) 기준으로 필요한 제한 시간 {need:.0f}초")
    n = lambda_async_attempts(p["lambda_async_default_retries"])
    print(f"Lambda 비동기 기본 실행 횟수: 최대 {n}번")
    ids = [f"req-{i}" for i in range(1, 11)]
    delivered = ids + ids[:3]
    print(f"10건에 중복 3건이 섞였을 때 부작용: 멱등 없음 {apply_effects(delivered, False)}번, 멱등 키 {apply_effects(delivered, True)}번")
    sends = [(0, "req-1"), (120, "req-1"), (299, "req-1"), (300, "req-1"), (310, "req-2")]
    kept = fifo_accepted(sends, p["fifo_dedup_window_seconds"])
    print(f"FIFO 5분 구간: 5번 보내서 {len(kept)}건 수락 {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
