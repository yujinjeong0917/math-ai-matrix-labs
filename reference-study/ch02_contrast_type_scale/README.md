# 2장 · 버튼 여러 개가 공유하는 규칙은 무엇인가

<!-- sync:start -->
> [디자이너의 시선으로 레퍼런스 읽기](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/reference-study) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [버튼 여러 개가 공유하는 규칙은 숫자로 어떻게 확인할까요?](https://math-ai-matrix-pi.vercel.app/tracks/reference-study/02-contrast-type-scale.html)

```bash
cd reference-study/ch02_contrast_type_scale
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `contrast.py` | 2장: 이 사이트 CSS 색의 WCAG 대비 비율과 글자 크기 비율. |
<!-- sync:end -->

이 사이트 CSS 색 토큰으로 WCAG 2.2의 대비 비율을 계산하고, 글자 크기 토큰의 이웃 비율을 냅니다. 식과 기준값은 W3C WCAG 2.2(https://www.w3.org/TR/WCAG22/)를 2026-11-18에 열어 확인했습니다. 파이썬 표준 라이브러리만 씁니다.

## 실행

```bash
python3 contrast.py
python3 -m unittest -q
```

## 기대 출력

```text
라이트 본문 ink / bg: 16.83:1  일반글자 AA 통과 / AAA 통과
라이트 보조 글자 ink-2 / bg: 5.07:1  일반글자 AA 통과 / AAA 미달
라이트 보조 글자 ink-2 / bg-alt: 4.66:1  일반글자 AA 통과 / AAA 미달
링크 link / bg: 5.57:1  일반글자 AA 통과 / AAA 미달
링크 link / bg-alt: 5.11:1  일반글자 AA 통과 / AAA 미달
기본 버튼 btn-ink / btn-bg: 5.57:1  일반글자 AA 통과 / AAA 미달
어두운 타일 on-dark / tile: 13.69:1  일반글자 AA 통과 / AAA 통과
어두운 타일 보조 on-dark-2 / tile: 5.80:1  일반글자 AA 통과 / AAA 미달
어두운 타일 링크 link-on-dark / tile: 4.94:1  일반글자 AA 통과 / AAA 미달
어두운 타일 카드 링크 link-on-dark / card-on-tile: 5.58:1  일반글자 AA 통과 / AAA 미달
다크 모드 보조 on-dark-2 / black: 8.16:1  일반글자 AA 통과 / AAA 통과
옛 보조색 #7a7a7a / bg (현재 조합 아님, CSS 주석이 말한 옛 값 4.3:1): 4.29:1  일반글자 AA 미달 / AAA 미달
# 아래 한 줄: 큰 글자(대비 기준 3:1이 적용되는 글자) 판정. 24px 이상 또는 굵은 글자 18.67px 이상이면 True
큰 글자 판정: 24px True / 19px False / 굵은 19px True / 굵은 18px False
리드(ink-2/bg) 폭 1440px(24px) 필요 기준 3:1 -> 5.07 True, 폭 390px(19px) 필요 기준 4.5:1 -> True
hero/display: 1.400
display/lead: 1.667
lead/tagline: 1.143
tagline/body: 1.235
body/caption: 1.214
caption/fine: 1.167
가장 큰/가장 작은: 4.67
```

## 이 실습이 확인하지 못하는 것

- 대비 비율은 접근성 기준의 한 항목입니다. 글자 크기, 굵기, 배경 이미지, 사용자 시력은 따로 봐야 해요.
- 안티에일리어싱과 투명도가 있는 색은 계산하지 않았습니다. 6자리 불투명 색만 썼습니다.
- "굵은 글자"는 WCAG에 숫자 굵기가 적혀 있지 않아 이 실습은 굵기 600 글자의 판정을 하지 않습니다.
