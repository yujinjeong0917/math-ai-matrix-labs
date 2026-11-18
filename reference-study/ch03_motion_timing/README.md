# 3장 · 클릭하면 무엇이 먼저 움직이나

<!-- sync:start -->
> [디자이너의 시선으로 레퍼런스 읽기](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/reference-study) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [버튼을 누르면 무엇이 얼마 동안 움직일까요?](https://math-ai-matrix-pi.vercel.app/tracks/reference-study/03-motion-timing.html)

```bash
cd reference-study/ch03_motion_timing
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `motion.py` | 3장: 지속 시간을 프레임 수로 바꾸고, ease 곡선에서 진행률을 읽는다. |
<!-- sync:end -->

CSS에 적힌 transition 선언(지속 시간, 이징)을 프레임 수와 진행률로 바꿉니다. 화면 녹화가 아니라 `assets/site.css` 값이며, 60Hz 주사율은 가정이고 30Hz와 120Hz는 비교용 가정입니다. ease 곡선은 W3C CSS Easing Functions Level 1(https://www.w3.org/TR/css-easing-1/)에서 2026-11-18에 확인한 값입니다. 파이썬 표준 라이브러리만 씁니다.

## 실행

```bash
python3 motion.py
python3 -m unittest -q
```

## 기대 출력

```text
선언 수 7 지속 시간별 {0.12: 6, 0.2: 1}
0.12초 = 60Hz에서 7.2프레임, 30Hz에서 3.6프레임, 120Hz에서 14.4프레임
0.2초 = 60Hz에서 12.0프레임, 30Hz에서 6.0프레임, 120Hz에서 24.0프레임
ease 진행률(시간 25/50/75%): [0.409, 0.802, 0.96]
곡선 위 s=0.7: 시간 0.5005, 진행률 0.8029 / 시간 50%의 진행률(4자리): 0.8024
linear 진행률(시간 25/50/75%): [0.25, 0.5, 0.75]
버튼 누름 크기(scale 0.95): [1.0, 0.9796, 0.9599, 0.952, 0.95]
44px 버튼이 줄어드는 높이: 2.2px (누른 뒤 41.8px)
움직임 줄이기 설정 선언이 없는 선택자: ['.pill', 'a.card', '.gnav__theme, .gnav__burger', '.mx-page .hero-action']
site.css 밖 확인: 인라인 <style>의 transition 0 개, @keyframes/animation 0 개
```

## 이 실습이 확인하지 못하는 것

- 실제 화면에서 움직임이 몇 ms에 시작하고 끝나는지는 재지 않았습니다. 브라우저와 기기마다 다를 수 있어요.
- 다른 서비스의 모션 값은 다루지 않습니다.
- 움직임을 줄이는 설정의 유무는 `site.css`에서 찾은 선언만 셌고, 자바스크립트나 다른 파일의 처리는 확인하지 않았습니다.
