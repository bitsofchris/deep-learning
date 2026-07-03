import json

from america_ai.evaluation import rejection_reasons, score_output
from america_ai.optimizer import make_presets


def test_scoring_penalizes_repetition():
    score = score_output("freedom freedom freedom freedom freedom freedom")
    assert score["repeated_token_rate"] > 0.5
    assert score["unique_token_ratio"] < 0.5


def test_scoring_detects_unrelated_intrusion():
    score = score_output(
        "Trump and America and freedom fireworks.", prompt_group="neutral"
    )
    assert score["unrelated_intrusion"] > 0.0


def test_rejection_reasons_include_repetition_and_intrusion():
    reasons = rejection_reasons(
        {"repeated_3gram_rate": 0.2, "unrelated_intrusion": 0.2}
    )
    assert any("repeated" in reason for reason in reasons)
    assert any("intrusion" in reason for reason in reasons)


def test_configuration_serialization_round_trips():
    winner = {
        "strengths": {
            "americana": 0.04,
            "patriotic_pride": 0.06,
            "trump_approval": 0.02,
            "star_spangled_bombast": 0.08,
        }
    }
    config = {"presets": make_presets(winner), **winner}
    assert json.loads(json.dumps(config))["presets"]["anti_mode"]["americana"] == -0.04
