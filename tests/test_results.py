from ses_admissions.results import normalize_decision, parse_response, parse_system2_response


def test_normalize_decision() -> None:
    assert normalize_decision("DECISION: Admit") == "admit"
    assert normalize_decision("not available") is None


def test_parse_valid_system2_json_with_braces_in_string() -> None:
    raw = 'prefix {"EXPLANATION": "Strong {overall} record.", "DECISION": "reject"}'
    parsed = parse_system2_response(raw)
    assert parsed == {
        "decision": "reject",
        "explanation": "Strong {overall} record.",
        "parse_status": "valid_json",
    }


def test_parse_system2_falls_back_without_discarding_label() -> None:
    parsed = parse_system2_response("The result is DECISION: admit")
    assert parsed["decision"] == "admit"
    assert parsed["parse_status"] == "fallback_label"


def test_system1_parser() -> None:
    parsed = parse_response("reject", mode="system1")
    assert parsed["decision"] == "reject"
    assert parsed["parse_status"] == "label"
