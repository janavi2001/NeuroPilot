import pytest

from interface.intent import IntentAdapter


def test_selected_cup_becomes_deterministic_instruction():
    assert IntentAdapter().convert({"selected_object": "CUP"}) == "Pick up the cup."


def test_intent_adapter_rejects_missing_target():
    with pytest.raises(ValueError):
        IntentAdapter().convert({})


def test_libero_adapter_is_explicit_about_cup_mapping():
    assert IntentAdapter().convert_libero("CUP") == (
        "Pick up the alphabet soup and place it in the basket."
    )
