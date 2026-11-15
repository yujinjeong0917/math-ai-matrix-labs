"""6장 라우터: 요청을 v1과 v2 중 어디로 보낼지 정한다.

route(user_id, p, salt): 해시 기반 고정 배정. 같은 사용자는 비율 p가 그대로인 한 늘 같은 버전을 본다.
  파이썬 내장 hash()는 프로세스마다 PYTHONHASHSEED로 값이 바뀌므로 쓰지 않고 hashlib(SHA-256)을 쓴다.
  p를 0.01 -> 0.05 -> 0.2로 올려도 이미 v2를 본 사용자는 계속 v2를 본다(버킷 값이 고정이라서).
route_random(rng, p): 요청마다 동전 던지기. 같은 사용자도 요청마다 버전이 바뀔 수 있다(비교용).
"""

import hashlib


def bucket(user_id, salt="release-ch06"):
    """사용자를 [0, 1) 사이의 고정된 값 하나로 바꾼다."""
    # salt와 user_id를 이어 SHA-256으로 해시한다. salt를 바꾸면 사용자의 버킷 배치가 새로 섞인다
    h = hashlib.sha256(f"{salt}:{user_id}".encode()).digest()
    # 앞 8바이트를 0 이상 2^64 미만 정수로 읽어 2^64로 나누면 [0, 1) 값. 거의 균등하게 퍼진다
    return int.from_bytes(h[:8], "big") / 2**64


def route(user_id, p, salt="release-ch06"):
    # 버킷 값이 p보다 작으면 v2. p를 올리면 기존 v2 사용자는 그대로 v2에 남고 새 사용자만 더해진다
    return "v2" if bucket(user_id, salt) < p else "v1"


def route_random(rng, p):
    return "v2" if rng.uniform() < p else "v1"
