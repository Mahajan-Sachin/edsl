import pytest
from edsl import Model, QuestionFreeText
from edsl.questions.question_thinking import QuestionThinking, thinking_question, THINKING_MAX_TOKENS
from edsl.language_models.output_token_policy import DEFAULT_OUTPUT_TOKENS


def test_default_model_max_tokens():
    model = Model()
    assert model.max_tokens == DEFAULT_OUTPUT_TOKENS
    assert model.max_tokens == 4096
    assert getattr(model, "_has_custom_token_limit", False) is False


def test_custom_model_max_tokens():
    model = Model("gpt-4o", max_tokens=2000)
    assert model.max_tokens == 2000
    assert getattr(model, "_has_custom_token_limit", False) is True

    model_param_dict = Model("gpt-4o", parameters={"max_tokens": 1500})
    assert model_param_dict.max_tokens == 1500
    assert getattr(model_param_dict, "_has_custom_token_limit", False) is True


def test_question_thinking_default_max_tokens():
    qt = QuestionThinking(question_name="t1", question_text="Think about this")
    assert qt._model.max_tokens == THINKING_MAX_TOKENS
    assert qt._model.max_tokens == 8192


def test_thinking_question_default_model_max_tokens():
    q = QuestionFreeText(question_name="q0", question_text="Describe entropy.")
    tq = thinking_question(q)
    assert tq._model.max_tokens == THINKING_MAX_TOKENS
    assert tq._model.max_tokens == 8192


def test_thinking_question_with_model_string():
    q = QuestionFreeText(question_name="q0", question_text="Describe entropy.")
    tq = thinking_question(q, model="gpt-4o")
    assert tq._model.max_tokens == 8192


def test_thinking_question_with_default_model_instance():
    q = QuestionFreeText(question_name="q0", question_text="Describe entropy.")
    m = Model("gpt-4o")
    assert m.max_tokens == 4096
    tq = thinking_question(q, model=m)
    assert tq._model.max_tokens == 8192
    # Ensure original model was not mutated
    assert m.max_tokens == 4096


def test_thinking_question_preserves_custom_model_max_tokens():
    q = QuestionFreeText(question_name="q0", question_text="Describe entropy.")
    m = Model("gpt-4o", max_tokens=2000)
    tq = thinking_question(q, model=m)
    assert tq._model.max_tokens == 2000


def test_thinking_question_with_test_model():
    q = QuestionFreeText(question_name="q0", question_text="Describe entropy.")
    tq = thinking_question(q, model="test")
    assert tq._model.model == "test"
    assert "max_tokens" not in tq._model.parameters


def test_thinking_question_serialization_roundtrip():
    q = QuestionFreeText(question_name="q0", question_text="Describe entropy.")
    tq = thinking_question(q)
    d = tq.to_dict()
    restored = QuestionFreeText.from_dict(d)
    assert restored._is_thinking_question is True
    assert restored._model.max_tokens == 8192

    # With custom max_tokens
    m_custom = Model("gpt-4o", max_tokens=1500)
    tq_custom = thinking_question(q, model=m_custom)
    d_custom = tq_custom.to_dict()
    restored_custom = QuestionFreeText.from_dict(d_custom)
    assert restored_custom._is_thinking_question is True
    assert restored_custom._model.max_tokens == 1500
