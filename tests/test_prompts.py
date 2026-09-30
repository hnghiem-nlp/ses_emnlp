from ses_admissions.constants import FROZEN_ACTIVITY_BLOCK
from ses_admissions.inference import chat_messages
from ses_admissions.prompts import attribute_order, build_prompt, build_system_prompt


SAMPLE_PROFILE = {
    "school_type": "Public",
    "gpa": 3.75,
    "sat": 1280,
    "activity": 5,
    "leadership": 2,
    "award": 1,
    "first_gen": "Yes",
    "fee_waiver": "No",
    "zip_code": "02138",
}


def test_activity_fields_remain_adjacent() -> None:
    for seed in (101, 211, 307):
        order = attribute_order(seed)
        start = order.index("activity")
        assert tuple(order[start : start + 3]) == FROZEN_ACTIVITY_BLOCK


def test_prompt_contains_only_presented_attributes() -> None:
    prompt = build_prompt(SAMPLE_PROFILE, mode="system1", prompt_seed=123, attribute_seed=101)
    assert "GPA: 3.75" in prompt
    assert "INCOME QUINTILE" not in prompt
    assert prompt.endswith("DECISION: ")


def test_system_prompt_conditions() -> None:
    omitted = build_system_prompt(institution="Example U", tier="t2", condition="omitted")
    specified = build_system_prompt(
        institution="Example U", tier="t2", condition="specified", acceptance_rate=24
    )
    assert "between 15% and 30%" in omitted
    assert "24%" in specified


def test_merge_system_prompt_for_models_without_system_role() -> None:
    merged = chat_messages("system", "user", merge_system=True)
    separate = chat_messages("system", "user", merge_system=False)
    assert merged == [{"role": "user", "content": "system user"}]
    assert [message["role"] for message in separate] == ["system", "user"]
