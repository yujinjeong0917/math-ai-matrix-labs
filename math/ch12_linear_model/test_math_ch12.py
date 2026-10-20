import importlib.util
from pathlib import Path

def load(name):
    file = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location("chapter_experiment", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_chapter_numbers():
    assert load("experiment.py").compute() == {'prediction': 2.0, 'gradient': -12.0, 'new_weight': 2.2, 'new_prediction': 4.4}

def test_torch_matches_numpy():
    assert load("experiment_torch.py").compute() == load("experiment.py").compute()
