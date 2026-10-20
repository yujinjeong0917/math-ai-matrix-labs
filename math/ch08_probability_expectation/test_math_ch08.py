import importlib.util
from pathlib import Path

def load(name):
    file = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location("chapter_experiment", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_chapter_numbers():
    assert load("experiment.py").compute() == {'probability_sum': 1.0, 'expectation': 1.0}

def test_torch_matches_numpy():
    assert load("experiment_torch.py").compute() == load("experiment.py").compute()
