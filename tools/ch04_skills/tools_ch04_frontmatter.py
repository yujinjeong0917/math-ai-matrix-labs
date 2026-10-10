"""SKILL.md 앞머리(frontmatter) 읽기와 규칙 검사.

YAML 전체를 다루지 않는다. SKILL.md 앞머리에 실제로 쓰이는 작은 부분만 읽는다.
- 첫 줄이 `---`, 다음 `---`까지가 앞머리
- `키: 값` 한 줄(값은 따옴표로 감쌀 수 있음)
- `키:` 다음 줄부터 두 칸 들여쓴 `하위키: 값` 묶음(metadata 같은 지도 하나)
표준 라이브러리만 쓴다.
"""

import re
from pathlib import Path

HERE = Path(__file__).parent
SKILLS_DIR = HERE / "data" / "skills"

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")  # 소문자·숫자, 하이픈은 사이에만 한 개씩
XML_RE = re.compile(r"<[^>]+>")


class FrontmatterError(ValueError):
    pass


def _unquote(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def split(text):
    """SKILL.md 글 전체 -> (앞머리 글, 본문 글). 앞머리가 없으면 오류."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise FrontmatterError("첫 줄이 --- 가 아니에요")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:])
    raise FrontmatterError("앞머리를 닫는 --- 가 없어요")


def parse(text):
    """앞머리 글 -> dict. 지도는 한 단계만 읽는다."""
    head, body = split(text)
    data, current = {}, None
    for raw in head.split("\n"):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("  ") and current is not None:
            k, sep, v = raw.strip().partition(":")
            if not sep:
                raise FrontmatterError(f"하위 줄에 ':'가 없어요: {raw!r}")
            data[current][k.strip()] = _unquote(v)
            continue
        k, sep, v = raw.partition(":")
        if not sep:
            raise FrontmatterError(f"':'가 없는 줄이에요: {raw!r}")
        k = k.strip()
        if v.strip() == "":
            data[k] = {}
            current = k
        else:
            data[k] = _unquote(v)
            current = None
    return data, body


def validate_spec(fm, dir_name=None):
    """agentskills.io 사양의 규칙. 어긴 규칙 목록을 돌려준다(빈 목록이면 통과)."""
    errs = []
    name = fm.get("name")
    if not isinstance(name, str) or not name:
        errs.append("name 없음")
    else:
        if len(name) > 64:
            errs.append("name 64자 초과")
        if not NAME_RE.match(name):
            errs.append("name 글자 규칙 위반(소문자·숫자·하이픈, 하이픈은 처음·끝·연속 불가)")
        if dir_name is not None and name != dir_name:
            errs.append("name이 폴더 이름과 다름")
    desc = fm.get("description")
    if not isinstance(desc, str) or not desc.strip():
        errs.append("description 없음")
    elif len(desc) > 1024:
        errs.append("description 1024자 초과")
    comp = fm.get("compatibility")
    if comp is not None and (not isinstance(comp, str) or not 1 <= len(comp) <= 500):
        errs.append("compatibility 1~500자 위반")
    meta = fm.get("metadata")
    if meta is not None and not isinstance(meta, dict):
        errs.append("metadata는 지도여야 함")
    return errs


def validate_anthropic(fm):
    """Anthropic 문서가 사양 위에 더 둔 규칙(예약어, XML 태그)."""
    errs = []
    name = fm.get("name") or ""
    desc = fm.get("description") or ""
    for w in ("anthropic", "claude"):
        if isinstance(name, str) and w in name:
            errs.append(f"name에 예약어 '{w}'")
    if isinstance(name, str) and XML_RE.search(name):
        errs.append("name에 XML 태그")
    if isinstance(desc, str) and XML_RE.search(desc):
        errs.append("description에 XML 태그")
    return errs


def listing_line(fm, limit=1536):
    """Claude Code의 스킬 목록 한 줄 흉내: description(+when_to_use)을 limit 글자에서 자른다."""
    text = fm.get("description", "")
    if fm.get("when_to_use"):
        text = text + " " + fm["when_to_use"]
    return f"{fm.get('name', '')}: {text[:limit]}", len(text) > limit


def load_skills(root=SKILLS_DIR):
    """폴더 안 스킬 -> [{dir, name, fm, body, frontmatter_text, files: {상대경로: 글}}] (이름 순)."""
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        text = (d / "SKILL.md").read_text(encoding="utf-8")
        fm, body = parse(text)
        head, _ = split(text)
        files = {}
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.name != "SKILL.md":
                files[f.relative_to(d).as_posix()] = f.read_text(encoding="utf-8")
        out.append({"dir": d.name, "name": fm.get("name"), "fm": fm, "body": body,
                    "frontmatter_text": head, "skill_md": text, "files": files})
    return out
