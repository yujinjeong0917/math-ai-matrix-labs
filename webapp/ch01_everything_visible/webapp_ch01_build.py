"""1인 개발 앱 출시 실전 1장: 실습용 편집기의 "빌드" 흉내.

editor/src/app.js(원본) -> editor/public/app.min.js(배포본) + app.min.js.map(지도 파일)

하는 일은 세 가지뿐이에요.
1. 주석 지우기, 줄 앞뒤 빈칸·빈 줄 지우기, 기호 둘레 빈칸 줄이기(압축, minify)
2. 정해 둔 지역 이름을 한 글자로 바꾸기(이름 바꾸기, 난독화의 가장 단순한 형태)
3. sourcesContent에 원본 전체를 담은 지도 파일 만들기

진짜 번들러가 아니에요. 문자열("...")과 주석만 구분하는 작은 토큰 분리기라서, 정규식 리터럴이나
템플릿 문자열(`...`)이 있는 코드는 다루지 못해요. 실습용 app.js는 그런 문법을 쓰지 않아요.
지도 파일의 mappings(줄·칸 대응표)는 비워 뒀어요. 이 장에서 보이려는 건 "원본 내용이 지도 파일에
통째로 들어 있다"는 점이라서예요.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "editor" / "src" / "app.js"
PUBLIC = HERE / "editor" / "public"
MIN_NAME = "app.min.js"
MAP_NAME = "app.min.js.map"

# 바꿀 이름(원래 이름 -> 짧은 이름). 속성 이름(.templates 같은)과 객체 키는 바꾸지 않는다.
RENAME = {
    "TEMPLATE_INDEX": "a", "GENERATE_ENDPOINT": "b", "currentTemplate": "c",
    "loadTemplateList": "d", "applyTemplate": "e", "renderCard": "f", "requestBackground": "g",
    "response": "h", "data": "i", "select": "j", "option": "k", "item": "l", "card": "m",
    "titleText": "n", "bodyText": "o", "heading": "p", "paragraph": "q", "userPrompt": "r",
    "templateFile": "s", "svgText": "t",
}

PUNCT = set("{}()[];,=+:<>!&|?*")


def tokenize(src):
    """(종류, 글자) 목록. 종류: str, comment, ident, space, newline, other."""
    out, i, n = [], 0, len(src)
    while i < n:
        ch = src[i]
        if ch in "\"'":
            j = i + 1
            while j < n and src[j] != ch:
                j += 2 if src[j] == "\\" else 1
            out.append(("str", src[i:j + 1]))
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j == -1 else j
            out.append(("comment", src[i:j]))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2) + 2
            out.append(("comment", src[i:j]))
            i = j
        elif ch == "\n":
            out.append(("newline", ch))
            i += 1
        elif ch in " \t\r":
            j = i
            while j < n and src[j] in " \t\r":
                j += 1
            out.append(("space", src[i:j]))
            i = j
        elif ch.isalpha() or ch in "_$":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_$"):
                j += 1
            out.append(("ident", src[i:j]))
            i = j
        else:
            out.append(("other", ch))
            i += 1
    return out


def minify(src, rename=True):
    toks = [t for t in tokenize(src) if t[0] != "comment"]
    # 이름 바꾸기: 앞이 '.'이 아닌 식별자만
    res = []
    for idx, (kind, text) in enumerate(toks):
        if kind == "ident" and rename and text in RENAME:
            prev = next((t for t in reversed(toks[:idx]) if t[0] not in ("space",)), None)
            if not (prev and prev[1] == "."):
                text = RENAME[text]
        res.append((kind, text))
    # 줄 단위로 다시 모으면서 빈칸 줄이기. 줄바꿈은 남겨 둔다(세미콜론 자동 삽입을 건드리지 않으려고).
    lines, cur = [], []
    for kind, text in res + [("newline", "\n")]:
        if kind == "newline":
            line = _squeeze(cur)
            if line:
                lines.append(line)
            cur = []
        else:
            cur.append((kind, text))
    return "\n".join(lines) + "\n"


def _squeeze(parts):
    out = []
    for i, (kind, text) in enumerate(parts):
        if kind == "space":
            prev = out[-1] if out else ""
            nxt = next((t[1] for t in parts[i + 1:] if t[0] != "space"), "")
            if not prev or not nxt or prev[-1] in PUNCT or nxt[0] in PUNCT:
                continue
            out.append(" ")
        else:
            out.append(text)
    return "".join(out).strip()


def string_literals(src):
    return [t[1] for t in tokenize(src) if t[0] == "str"]


def comments(src):
    return [t[1] for t in tokenize(src) if t[0] == "comment"]


def source_map(original, names):
    return {
        "version": 3,
        "file": MIN_NAME,
        "sources": ["../src/app.js"],
        "sourcesContent": [original],
        "names": sorted(names),
        "mappings": "",
    }


def build(write=True):
    original = SRC.read_text(encoding="utf-8")
    minified = minify(original) + f"//# sourceMappingURL={MAP_NAME}\n"
    smap = json.dumps(source_map(original, RENAME.keys()), ensure_ascii=False, indent=1) + "\n"
    if write:
        (PUBLIC / MIN_NAME).write_text(minified, encoding="utf-8", newline="\n")
        (PUBLIC / MAP_NAME).write_text(smap, encoding="utf-8", newline="\n")
    return original, minified, smap



if __name__ == "__main__":
    o, m, s = build()
    print(f"원본 {len(o.encode())}바이트 -> 배포본 {len(m.encode())}바이트, 지도 {len(s.encode())}바이트")
