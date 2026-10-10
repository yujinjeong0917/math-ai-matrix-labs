"""6장 라우터: 요청을 v1과 v2 중 어디로 보낼지 정한다.

route(user_id, p, salt): 해시 기반 고정 배정. 같은 사용자는 비율 p가 그대로인 한 늘 같은 버전을 본다.
  파이썬 내장 hash()는 프로세스마다 PYTHONHASHSEED로 값이 바뀌므로 쓰지 않고 hashlib(SHA-256)을 쓴다.
  p를 0.01 -> 0.05 -> 0.2로 올려도 이미 v2를 본 사용자는 계속 v2를 본다(버킷 값이 고정이라서).
route_random(rng, p): 요청마다 동전 던지기. 같은 사용자도 요청마다 버전이 바뀔 수 있다(비교용).
"""

import hashlib


def bucket(user_id, salt="release-ch06"):
    """사용자를 [0, 1) 사이의 고정된 값 하나로 바꾼다."""
    h = hashlib.sha256(f"{salt}:{user_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def route(user_id, p, salt="release-ch06"):
    return "v2" if bucket(user_id, salt) < p else "v1"


def route_random(rng, p):
    return "v2" if rng.uniform() < p else "v1"
