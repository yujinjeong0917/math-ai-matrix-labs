"""README의 '장 목록' 표를 지금 커밋된 장 폴더로 다시 만든다.
웹 링크는 사이트 저장소(../math-ai-matrix)에 커밋된 장만 단다. 사용: python3 scripts/update_readme.py"""
import os, re, subprocess
SITE = "https://math-ai-matrix-pi.vercel.app/tracks/"
NAMES = {"llm": "밑바닥부터 만드는 LLM", "rl": "밑바닥부터 만드는 강화학습", "eval": "모델 평가",
         "algorithms": "그림으로 익히는 알고리즘", "pipelines": "ML 파이프라인 구축", "productivity": "AX 에세이", "tools": "AI 도구 실전", "webapp": "1인 개발 앱 출시 실전"}
ORDER = list(NAMES)
files = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
chs = sorted({"/".join(f.split("/")[:2]) for f in files if re.match(r"^(%s)/ch" % "|".join(ORDER), f)},
             key=lambda c: (ORDER.index(c.split("/")[0]), c))
site_files = subprocess.run(["git", "-C", "../math-ai-matrix", "ls-files", "tracks"], capture_output=True, text=True).stdout.split()
web = {}
for f in site_files:
    m = re.match(r"tracks/([a-z]+)/(\d\d|bridge)-.*\.html$", f)
    if m: web[(m.group(1), int(m.group(2)) if m.group(2).isdigit() else "bridge")] = f[len("tracks/"):]
rows = []
for c in chs:
    d, ch = c.split("/")
    m = re.match(r"ch(\d+)", ch)
    n = int(m.group(1)) if m else ("bridge" if ch.startswith("chbridge") else None)
    label = f"{NAMES[d]} {n}장" if isinstance(n, int) else (f"{NAMES[d]} 연결 장" if n == "bridge" else f"{NAMES[d]} {ch}")
    w = web.get((d, n))
    rows.append(f"| `{c}/` | {label} | " + (f"[웹 챕터]({SITE}{w})" if w else "웹 챕터 공개 준비 중") + " |")
p = "README.md"; s = open(p, encoding="utf-8").read()
s = re.sub(r"(\| 폴더 \| 장 \| 웹 \|\n\|---\|---\|---\|\n)(?:\|.*\|\n)+", lambda m: m.group(1) + "\n".join(rows) + "\n", s)
open(p, "w", encoding="utf-8").write(s)
print(f"장 {len(rows)}개")
