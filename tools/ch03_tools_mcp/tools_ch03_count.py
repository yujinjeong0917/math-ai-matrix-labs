"""AI 도구 실전 3장: 앱 N개와 도구 M개를 잇는 연결 수.

- 표준 없이 짝마다 따로 이으면 연결 코드는 N × M개(앱과 도구의 모든 짝).
- 공통 규약(MCP)에 맞추면 앱마다 클라이언트 1개, 도구마다 서버 1개라 N + M개.
- 둘의 차이: N × M − (N + M) = (N − 1)(M − 1) − 1.
"""

from itertools import product


def pairwise(n, m):
    return n * m


def standard(n, m):
    return n + m


def gap(n, m):
    return (n - 1) * (m - 1) - 1


def pairwise_by_listing(apps, tools):
    """짝을 하나씩 늘어놓아 센다(공식과 대조용)."""
    return len(list(product(apps, tools)))


def standard_by_listing(apps, tools):
    return len([("client", a) for a in apps] + [("server", t) for t in tools])


def ripple(n, m):
    """한쪽이 바뀌면 고칠 연결 코드 수.

    도구 하나가 인자 이름을 바꾸면: 짝마다 이은 방식은 그 도구에 붙은 앱 N곳, 표준은 그 도구의 서버 1곳.
    앱 하나가 바뀌면(새 앱 추가 포함): 짝마다 이은 방식은 도구 M개, 표준은 그 앱의 클라이언트 1개.
    """
    return {"tool_changes": {"pairwise": n, "standard": 1},
            "app_changes": {"pairwise": m, "standard": 1},
            "add_tool": {"pairwise": n, "standard": 1},
            "add_app": {"pairwise": m, "standard": 1}}


def table(max_n=6, max_m=6):
    return [{"n": n, "m": m, "pairwise": pairwise(n, m), "standard": standard(n, m), "gap": gap(n, m)}
            for n in range(1, max_n + 1) for m in range(1, max_m + 1)]
