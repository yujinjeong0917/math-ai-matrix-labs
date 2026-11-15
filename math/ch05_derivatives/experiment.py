"""5장 · 미분: 손계산과 같은 숫자를 만드는 결정적 실험.

무엇을 재는가: x^2의 x = 3에서의 미분(순간 변화율)을 유한 차분으로 어림한 값과 정확한 값 6을 비교한다.
입력 없음. h = 0.001, 기준점 x = 3이 고정돼 있다.
출력: {"finite_difference": 6.001, "exact": 6.0}. 장 본문 숫자와의 대응은 README로 확인 필요.
실행하면 결과를 results/ch05.json에 저장한다. 이 파일은 numpy가 필요하다.
"""
from pathlib import Path
import json
import numpy as np

def compute():
    # 유한 차분 ((3+h)^2 - 3^2) / h = 6 + h = 6.001. h가 작을수록 정확한 미분 2x = 6에 가까워진다
    # 소수 셋째 자리까지 반올림해 부동소수점 잔여 오차를 숨긴다
    h=0.001; derivative=((3+h)**2-3**2)/h
    return {'finite_difference': round(derivative,3), 'exact': 6.0}

if __name__ == "__main__":
    result = compute()
    # 같은 폴더의 results/에 JSON으로 저장한다(폴더가 없으면 만든다). 사이트 본문 숫자와 대조하는 재현 기록
    target = Path(__file__).with_name("results") / "ch05.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
