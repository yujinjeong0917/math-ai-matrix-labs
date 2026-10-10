"""1인 개발 앱 출시 실전 2장: 환경 변수 값을 코드에 그대로 써 넣는 빌드 흉내.

Vite 같은 빌드 도구는 `import.meta.env.VITE_이름`을 빌드할 때 그 값 문자열로 바꿔 넣어요.
VITE_로 시작하지 않는 이름은 브라우저용 코드에 들어가지 않아요(여기서는 undefined로 바꿔요).
진짜 Vite가 아니라 그 규칙 하나만 흉내 내요.

project/env.good    + src/app.js          -> project/dist_good/app.js, app.js.map
project/env.mistake + src/app_mistake.js  -> project/dist_mistake/app.js, app.js.map
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
PROJECT = HERE / "project"
PUBLIC_PREFIX = "VITE_"
REF = re.compile(r"import\.meta\.env\.([A-Za-z_][A-Za-z0-9_]*)")

BUILDS = {
    "good": ("env.good", "src/app.js", "dist_good"),
    "mistake": ("env.mistake", "src/app_mistake.js", "dist_mistake"),
}


def read_env(path):
    """KEY=VALUE 줄만 읽어요. #으로 시작하는 줄과 빈 줄은 건너뛰어요."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def inline_env(src, env):
    """VITE_ 이름은 값 문자열로, 나머지는 undefined로 바꿔요. 바꾼 이름 목록도 돌려줘요."""
    inlined, hidden = [], []

    def sub(m):
        name = m.group(1)
        if name.startswith(PUBLIC_PREFIX) and name in env:
            inlined.append(name)
            return json.dumps(env[name])
        hidden.append(name)
        return "undefined"

    return REF.sub(sub, src), inlined, hidden


def build(which, write=True):
    env_name, src_name, dist_name = BUILDS[which]
    env = read_env(PROJECT / env_name)
    src = (PROJECT / src_name).read_text(encoding="utf-8")
    out, inlined, hidden = inline_env(src, env)
    out += "//# sourceMappingURL=app.js.map\n"
    smap = json.dumps({"version": 3, "file": "app.js", "sources": ["../" + src_name],
                       "sourcesContent": [src], "names": [], "mappings": ""},
                      ensure_ascii=False, indent=1) + "\n"
    if write:
        dist = PROJECT / dist_name
        dist.mkdir(exist_ok=True)
        (dist / "app.js").write_text(out, encoding="utf-8", newline="\n")
        (dist / "app.js.map").write_text(smap, encoding="utf-8", newline="\n")
    return {"env": env, "src": src, "out": out, "map": smap, "inlined": inlined, "hidden": hidden}


if __name__ == "__main__":
    for w in BUILDS:
        r = build(w)
        print(w, "브라우저 코드에 들어간 이름:", r["inlined"], "/ 빠진 이름:", r["hidden"])
