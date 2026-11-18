"""장 폴더마다 README 머리 블록(웹 챕터 링크, 실행 명령, 파일 안내)을 맞춘다.
- README가 없으면 만든다. 있으면 `<!-- sync:start -->`~`<!-- sync:end -->` 블록만 갱신하고 본문은 건드리지 않는다.
- 웹 링크는 사이트 저장소(../math-ai-matrix/tracks)의 실제 파일에서 찾는다.
사용: python3 scripts/sync_chapter_readmes.py [--check]"""
import ast, html, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT.parent / "math-ai-matrix" / "tracks"
SITE = "https://math-ai-matrix-pi.vercel.app/tracks/"
REPO = "https://github.com/yujinjeong0917/math-ai-matrix-labs"
START, END = "<!-- sync:start -->", "<!-- sync:end -->"
NAMES = {"llm": "밑바닥부터 만드는 LLM", "rl": "밑바닥부터 만드는 강화학습", "eval": "모델 평가",
         "algorithms": "그림으로 익히는 알고리즘", "pipelines": "ML 파이프라인 구축", "productivity": "협업 도구가 어긋날 때",
         "tools": "AI 도구 실전", "webapp": "1인 개발 앱 출시 실전", "finetune": "내 데이터로 LLM 파인튜닝하기",
         "math": "AI를 읽기 위한 수학", "aws": "AWS 배포·운영", "or": "경영과학", "causal": "인과추론", "agent-build": "에이전트 만들기",
         "waiting-ux": "기다림을 설계하다", "reference-study": "디자이너의 시선으로 레퍼런스 읽기"}


def web_page(track: str, folder: str):
    """chNN_슬러그 폴더에 대응하는 사이트 장 파일을 찾는다(번호 기준)."""
    m = re.match(r"ch(\d+)", folder)
    key = m.group(1).zfill(2) if m else "bridge"
    hits = sorted((SITE_DIR / track).glob(f"{key}-*.html")) if (SITE_DIR / track).is_dir() else []
    return hits[0] if hits else None


def page_title(p: Path) -> str:
    t = re.search(r"<title>([^<]*)</title>", p.read_text(encoding="utf-8"))
    return html.unescape(t.group(1).split(" · ")[0].strip()) if t else p.stem


def py_files(d: Path):
    return sorted(f for f in d.glob("*.py") if not f.name.startswith("test_"))


def needs_heavy_deps(d: Path) -> bool:
    for f in d.glob("*.py"):
        if re.search(r"^\s*(import|from)\s+(numpy|torch|matplotlib|pytest)\b", f.read_text(encoding="utf-8"), re.M):
            return True
    return False


def test_runner(d: Path) -> str:
    """테스트가 unittest 클래스면 'unittest', 아니면(pytest 함수·fixture) 'pytest'. 무거운 의존성이 있어도 pytest."""
    if needs_heavy_deps(d):
        return "pytest"
    tests = list(d.glob("test_*.py"))
    return "unittest" if tests and all("unittest.TestCase" in t.read_text(encoding="utf-8") for t in tests) else "pytest"


def first_doc_line(f: Path) -> str:
    try:
        doc = ast.get_docstring(ast.parse(f.read_text(encoding="utf-8")))
    except SyntaxError:
        doc = None
    return doc.strip().splitlines()[0].strip() if doc else ""


def block(track: str, d: Path) -> str:
    page = web_page(track, d.name)
    head = [START, f"> [{NAMES[track]}]({REPO}/tree/main/{track}) 트랙 · 전체 목차는 [저장소 README]({REPO})"]
    head.append(f"> 웹 챕터: [{page_title(page)}]({SITE}{track}/{page.name})" if page else "> 웹 챕터: 공개 준비 중")
    cmd = (f"uv sync   # 저장소 루트에서 한 번\nuv run pytest -q {d.relative_to(ROOT)}" if test_runner(d) == "pytest"
           else f"cd {d.relative_to(ROOT)}\npython3 -m unittest -q   # 표준 라이브러리만 사용")
    rows = [f"| `{f.name}` | {first_doc_line(f) or '-'} |" for f in py_files(d)]
    files = "\n".join(["| 파일 | 하는 일 |", "|---|---|", *rows]) if rows else ""
    out = head + ["", "```bash", cmd, "```"] + (["", files] if files else []) + [END]
    return "\n".join(out)


def sync(check: bool) -> int:
    changed = 0
    for tdir in sorted(ROOT.iterdir()):
        if tdir.name not in NAMES:
            continue
        for d in sorted(tdir.glob("ch*")):
            if not d.is_dir():
                continue
            p, b = d / "README.md", block(tdir.name, d)
            if p.exists():
                s = p.read_text(encoding="utf-8")
                if START in s:
                    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: b, s, flags=re.S)
                else:
                    lines = s.split("\n")
                    i = next((k for k, l in enumerate(lines) if l.startswith("# ")), -1)
                    lines[i + 1:i + 1] = ["", b]
                    new = "\n".join(lines)
            else:
                page = web_page(tdir.name, d.name)
                title = page_title(page) if page else d.name
                new = f"# {title}\n\n{b}\n"
            if new != (p.read_text(encoding="utf-8") if p.exists() else ""):
                changed += 1
                if not check:
                    p.write_text(new, encoding="utf-8")
    print(f"갱신 필요 {changed}개" if check else f"갱신 {changed}개")
    return 1 if (check and changed) else 0


if __name__ == "__main__":
    sys.exit(sync("--check" in sys.argv))
