import pytest

from edsl.inference_services.inference_service_registry import InferenceServiceRegistry
from edsl.inference_services.services.groq_service import GroqService


def test_groq_service_is_registered():
    registry = InferenceServiceRegistry()
    assert "groq" in registry.list_registered_services()


def test_groq_async_client_instantiation():
    groq = pytest.importorskip("groq")
    GroqService._async_client_instances = {}
    client = GroqService.async_client(api_key="gsk_test_mock_key")
    assert isinstance(client, groq.AsyncGroq)
    # Ensure it's not using OpenAI's DefaultAioHttpClient
    assert client._client.__class__.__module__ != "openai._base_client"
