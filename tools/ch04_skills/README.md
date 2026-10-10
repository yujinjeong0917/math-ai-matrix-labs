# 4장. 같은 지시를 매번 붙여넣지 않으려면 (`ch04_skills`)

웹 챕터: `math-ai-matrix/tracks/tools/04-skills.html`

관통 프로젝트(가상의 작은 팀이 만드는 "주간 고객 문의 요약 보고서")에서, 점점 길어진 작업 지시를 매번 붙여넣는 대신 프로젝트 지침 파일(AGENTS.md)과 스킬(SKILL.md)로 나눴을 때 대화마다 실리는 토큰이 어떻게 달라지는지를 네트워크·API 키 없이 재현한다.

1. 직접 만든 스킬 6개(`data/skills/`)의 앞머리(frontmatter)를 작은 파서로 읽고, Agent Skills 사양과 Anthropic 문서의 규칙으로 검사한다.
2. 글자 수 근사 규칙(가정값, `data/token_rule.json`)으로 파일마다 토큰을 세고, 한 주 대화 15번에서 "전부 싣기"와 "앞머리만 + 쓰는 것만 본문"(단계적 공개)을 비교한다.
3. 스킬 수 N을 늘리며 닫힌 식과 비교하고, 파일을 읽을 때마다 모델 호출이 한 번 더 느는 비용(2장 캐싱 배율 포함)을 잰다.

## 실행
```bash
uv run python tools/ch04_skills/experiments.py   # results/ch04.json 갱신
uv run pytest -q tools/ch04_skills
```
표준 라이브러리만 쓴다(fractions로 정확히 계산). 실제 모델을 부르지 않는다.

## 파일
| 파일 | 역할 |
|---|---|
| `tools_ch04_frontmatter.py` | SKILL.md 앞머리 파서(YAML의 작은 부분만), 사양·Anthropic 규칙 검사, Claude Code 목록 줄 자르기 흉내 |
| `tools_ch04_budget.py` | 글자 수 근사 토큰, 전부 싣기 / 단계적 공개 비교, 스킬 수 N 확장, 호출 단위 비용 |
| `experiments.py` | E0~E8 |
| `data/skills/` | 직접 만든 스킬 6개. `weekly-inquiry-report`는 딸린 파일 2개와 스크립트 1개를 가진다 |
| `data/project/AGENTS.md` | 직접 만든 프로젝트 지침 예시(늘 실리는 쪽) |
| `data/token_rule.json` | 토큰 근사 규칙(가정값)과 민감도용 대안 두 개 |
| `data/week_tasks.json` | 한 주 대화 15번에서 어떤 스킬을 쓰고 어떤 파일을 읽었는지(가정값), 호출 단위 비용의 가정값 |
| `data/vendors_2026-10-10.json` | 도구별 규칙(이름 64자, 설명 1,024자, 목록 1,536자, Codex 32 KiB 등). 항목마다 출처 URL과 확인일 |
| `checks/ch04.csv` | 독자 기록 양식(예시 행은 모형 값) |

## 한계
- 토큰 수는 실제 토크나이저가 아니라 정한 규칙으로 센 근사값이다. 규칙을 바꿔도 절감 비율은 거의 그대로였지만(E4), 절대값은 크게 달라진다. 실제 값은 응답의 usage로 확인한다.
- 모델이 어떤 스킬을 고를지는 모델이 description을 보고 판단한다. 이 실습은 그 판단을 흉내 내지 않고, 고른 결과를 `week_tasks.json`에 가정으로 둔다.
- 앞머리 파서는 SKILL.md에 흔히 쓰는 모양(한 줄 값, 한 단계 지도)만 읽는다. 여러 줄 문자열(`|`, `>`)과 목록은 읽지 않는다.
- 호출 단위 비용은 "파일 하나를 읽을 때마다 호출 1번"(또는 묶어 읽기)으로 단순화했다. 실제 도구는 한 번에 여러 파일을 읽기도 한다.

확인일 2026-10-10.
