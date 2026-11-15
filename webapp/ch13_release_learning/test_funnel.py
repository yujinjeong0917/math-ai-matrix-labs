"""funnel.analyze 테스트. 확인하는 것: 최대 이탈 단계 선택, 단계 수가 늘어나는 입력 거부."""

import pytest
from funnel import analyze

# 전환율: open 0.6, start 0.5, input 0.3, export 0.6 -> input이 최저. 90/300 = 0.3
def test_finds_largest_step_drop():
    result = analyze([("install", 1000), ("open", 600), ("start", 300), ("input", 90), ("export", 54)])
    assert result["investigate_first"] == "input"
    assert result["rows"][3]["step_rate"] == pytest.approx(.3)

# 뒤 단계(12)가 앞 단계(10)보다 크면 ValueError
def test_rejects_mixed_or_broken_counts():
    with pytest.raises(ValueError):
        analyze([("start", 10), ("finish", 12)])
