"""LLM integration with OpenRouter and DSPy."""

from .openrouter import OpenRouterClient, StreamingResponse, get_client
from .signatures import SignatureBuilder, DynamicSignature, Field, get_signature
from .dspy_integration import DSPyModule, create_module

__all__ = [
    "OpenRouterClient",
    "StreamingResponse",
    "get_client",
    "SignatureBuilder",
    "DynamicSignature",
    "Field",
    "get_signature",
    "DSPyModule",
    "create_module",
]
