"""루트 README와 트랙별 README(트랙 목차)를 장 폴더 기준으로 다시 만든다.
웹 링크는 사이트 저장소(../math-ai-matrix/tracks)에 있는 장만 단다.
사용: python3 scripts/update_readme.py && python3 scripts/sync_chapter_readmes.py"""
import re
from pathlib import Path
import sync_chapter_readmes as S

ROOT, NAMES, SITE, REPO = S.ROOT, S.NAMES, S.SITE, S.REPO
ORDER = list(NAMES)
HEAVY = {"llm", "rl", "eval", "algorithms", "pipelines", "tools", "webapp", "finetune", "math", "productivity"}


def chapters(track):
    return [d for d in sorted((ROOT / track).glob("ch*")) if d.is_dir()]


def rows(track):
    out = []
    for d in chapters(track):
        page = S.web_page(track, d.name)
        title = S.page_title(page) if page else d.name
        link = f"[웹 챕터]({SITE}{track}/{page.name})" if page else "공개 준비 중"
        out.append(f"| [`{d.name}/`]({d.name}/) | {title} | {link} |")
    return out


def heavy_count(track):
    """uv 환경(numpy·torch·pytest)이 필요한 장 수."""
    return sum(S.test_runner(d) == "pytest" for d in chapters(track))


def uses_heavy(track):
    return heavy_count(track) > 0


def track_url(track):
    """트랙 첫 장 페이지(사이트에는 트랙별 index가 없다). 없으면 트랙 목록 페이지."""
    hits = sorted((S.SITE_DIR / track).glob("*.html")) if (S.SITE_DIR / track).is_dir() else []
    return f"{SITE}{track}/{hits[0].name}" if hits else SITE


def dep_label(track):
    h, n = heavy_count(track), len(chapters(track))
    return "표준 라이브러리만" if h == 0 else (f"`uv` 환경 필요(numpy·torch·pytest)" if h == n else f"{h}/{n}장만 `uv` 환경 필요")


def track_readme(track):
    n = len(chapters(track))
    run = ("```bash\nuv sync   # 저장소 루트에서 한 번\nuv run pytest -q " + track + "   # " + dep_label(track) + "\n```" if uses_heavy(track)
           else "```bash\ncd " + track + "/ch00_...   # 장 폴더로 이동\npython3 -m unittest -q   # 표준 라이브러리만 사용\n```")
    return "\n".join([f"# {NAMES[track]}", "",
        f"[저장소 전체 목차]({REPO}) · [웹에서 이 트랙 보기]({track_url(track)}) · 장 {n}개", "",
        "## 실행", "", run, "", "## 장 목록", "", "| 폴더 | 웹 챕터 제목 | 웹 |", "|---|---|---|", *rows(track), ""])


def root_readme():
    summary = []
    for t in ORDER:
        summary.append(f"| [`{t}/`]({t}/) | {NAMES[t]} | {len(chapters(t))} | {dep_label(t)} | [웹 트랙]({track_url(t)}) |")
    parts = [
        "# math-ai-matrix-labs", "",
        f"[수학 × AI 매트릭스]({SITE.rsplit('/tracks/', 1)[0]}) 학습 트랙의 실습 코드예요. 웹 챕터에서 계산한 숫자를 직접 다시 만들어 볼 수 있게, 장마다 폴더 하나에 코드·테스트·README를 두었어요. 데이터는 모두 코드 안에서 만드는 합성 데이터라 같은 시드에서는 같은 결과가 나와요.", "",
        "## 어디서 시작하나요", "",
        "1. 웹 챕터를 읽다가 \"실습 코드\" 링크를 누르면 그 장 폴더로 와요. 폴더 README의 `웹 챕터` 링크로 되돌아갈 수 있어요.",
        "2. 트랙 전체를 훑으려면 아래 표에서 트랙 폴더를 열어요. 트랙 폴더 README가 장 목록과 웹 링크를 모아 둬요.", "",
        "## 트랙 한눈에", "",
        "| 트랙 폴더 | 이름 | 장 수 | 필요한 것 | 웹 |", "|---|---|---|---|---|", *summary, "",
        "## 실행", "",
        "```bash",
        "# `uv` 환경(numpy·torch·pytest)이 필요한 장이 있는 트랙: " + ", ".join(t for t in ORDER if uses_heavy(t)),
        "uv sync                      # Python 3.12, 버전은 uv.lock으로 고정",
        "uv run python scripts/run_all_tests.py   # 모든 장 테스트를 장 폴더마다 따로 실행",
        "",
        "# 표준 라이브러리만 쓰는 트랙: " + ", ".join(t for t in ORDER if not uses_heavy(t)) + " (설치 없이 장 폴더에서)",
        "cd aws/ch11_retry_duplicates && python3 -m unittest -q",
        "python3 scripts/run_all_tests.py aws or causal agent-build   # 이 트랙들만 한 번에",
        "```", "",
        "## 폴더마다 같은 구성", "",
        "- `README.md`: 웹 챕터 링크, 실행 명령, 파일 안내(머리 블록은 `scripts/sync_chapter_readmes.py`가 갱신해요).",
        "- `*_np.py` / `*_torch.py`: NumPy 최소 구현과 PyTorch 대응 구현(해당 장만).",
        "- `experiments.py`: 비교 실험. 결과는 `results/*.json`에 저장해요.",
        "- `test_*.py`: 수치 일치와 수식 성질 검증.", "",
        "## 이 저장소가 확인하지 못하는 것", "",
        "- 실측 시간은 기기마다 달라요. 계산량과 테스트 결과는 같아야 해요.",
        "- AWS·외부 서비스 장(`aws`, `pipelines` 일부)은 계정 없이 도는 계산·검증 코드예요. 실제 서비스 호출, 요금, 한도는 측정하지 않아요.", "",
        "## 이 README를 다시 만드는 법", "",
        "```bash",
        "python3 scripts/update_readme.py          # 루트·트랙 README",
        "python3 scripts/sync_chapter_readmes.py   # 장 README 머리 블록",
        "```", ""]
    return "\n".join(parts)


if __name__ == "__main__":
    (ROOT / "README.md").write_text(root_readme(), encoding="utf-8")
    for t in ORDER:
        (ROOT / t / "README.md").write_text(track_readme(t), encoding="utf-8")
    print("루트 1 + 트랙", len(ORDER))
