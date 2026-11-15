"""5장 테스트(pytest 형식). 두 가지를 확인한다.
1) numpy 실험의 결과가 손계산 기대값과 정확히 같다(유한 차분 6.001(= 6 + h), 정확한 미분 6.0).
2) PyTorch 실험의 결과가 numpy 실험과 정확히 같다(두 구현의 교차 검증).
두 파일을 모듈로 불러 오므로 __main__ 블록은 실행되지 않고 results/ 파일은 쓰지 않는다.
실행에는 numpy와 torch가 모두 필요하다."""
import importlib.util
from pathlib import Path

def load(name):
    file = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location("chapter_experiment", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_chapter_numbers():
    # 기대값 딕셔너리는 손으로 계산해 확인할 수 있는 값: 유한 차분 6.001(= 6 + h), 정확한 미분 6.0
    assert load("experiment.py").compute() == {'finite_difference': 6.001, 'exact': 6.0}

def test_torch_matches_numpy():
    # 같은 연산을 두 라이브러리로 돌려 결과가 일치하는지 본다. 허용 오차 없이 == 로 비교하므로, 반올림되거나 정확히 표현되는 값만 쓴다
    assert load("experiment_torch.py").compute() == load("experiment.py").compute()
