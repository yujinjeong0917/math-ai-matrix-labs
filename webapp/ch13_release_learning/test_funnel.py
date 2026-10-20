import pytest
from funnel import analyze

def test_finds_largest_step_drop():
    result = analyze([("install", 1000), ("open", 600), ("start", 300), ("input", 90), ("export", 54)])
    assert result["investigate_first"] == "input"
    assert result["rows"][3]["step_rate"] == pytest.approx(.3)

def test_rejects_mixed_or_broken_counts():
    with pytest.raises(ValueError):
        analyze([("start", 10), ("finish", 12)])
