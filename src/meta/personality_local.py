"""Local model integration for personality system.

Supports:
- Ollama (llama, mistral, etc.)
- vLLM
- llama.cpp via llama-cpp-python
- HuggingFace transformers

Features:
- Personality-aware generation with logprobs
- Personality embeddings for similarity matching
- Model-specific voice calibration
- Fine-tuning data generation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .personality import (
    Personality,
    TraitDimension,
    TraitProfile,
    VoicePattern,
    ToneRegister,
    ResponseLength,
    PERSONALITY_ARCHETYPES,
    get_personality_manager,
)
from .personality_logprobs import (
    ResponseLogprobs,
    TokenLogprob,
    LogprobPersonalityAnalyzer,
    PersonalityProber,
    PersonalitySteering,
    parse_openai_logprobs,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Local Model Backend Types
# ─────────────────────────────────────────────────────────────────────────────


class LocalModelBackend(str, Enum):
    """Supported local model backends."""
    OLLAMA = "ollama"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    TRANSFORMERS = "transformers"
    MLXLM = "mlx-lm"  # Apple Silicon optimized


@dataclass
class LocalModelConfig:
    """Configuration for local model."""
    backend: LocalModelBackend
    model_name: str

    # Connection settings
    base_url: str = "http://localhost:11434"  # Default Ollama
    api_key: Optional[str] = None

    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048

    # Logprob settings
    logprobs: bool = True
    top_logprobs: int = 5

    # Model-specific
    context_length: int = 4096
    num_gpu_layers: int = -1  # -1 = all layers on GPU

    # Personality integration
    personality_strength: float = 1.0
    use_steering_biases: bool = True


@dataclass
class LocalGenerationResult:
    """Result from local model generation."""
    text: str
    logprobs: Optional[ResponseLogprobs] = None
    tokens_generated: int = 0
    generation_time: float = 0.0
    model: str = ""
    personality_analysis: Optional[Dict[str, Any]] = None

    # Token-level details
    token_times: List[float] = field(default_factory=list)
    tokens_per_second: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Local Model Client
# ─────────────────────────────────────────────────────────────────────────────


class LocalModelClient(ABC):
    """Abstract base for local model clients."""

    def __init__(self, config: LocalModelConfig, personality: Optional[Personality] = None):
        self.config = config
        self.personality = personality
        self._analyzer: Optional[LogprobPersonalityAnalyzer] = None
        self._steering: Optional[PersonalitySteering] = None

        if personality:
            self._analyzer = LogprobPersonalityAnalyzer(personality)
            self._steering = PersonalitySteering(personality)

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> LocalGenerationResult:
        """Generate text with optional logprobs."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generation."""
        pass

    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts."""
        pass

    def set_personality(self, personality: Personality):
        """Set or update personality."""
        self.personality = personality
        self._analyzer = LogprobPersonalityAnalyzer(personality)
        self._steering = PersonalitySteering(personality)

    def get_personality_system_prompt(self) -> str:
        """Get system prompt for personality."""
        if self.personality:
            return self.personality.generate_system_prompt()
        return ""

    def get_steering_biases(self, context_type: str = "general") -> Dict[str, float]:
        """Get logit biases for personality steering."""
        if self._steering and self.config.use_steering_biases:
            return self._steering.generate_biases(self.config.personality_strength)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Client
# ─────────────────────────────────────────────────────────────────────────────


class OllamaClient(LocalModelClient):
    """Client for Ollama local models."""

    def __init__(
        self,
        config: LocalModelConfig,
        personality: Optional[Personality] = None,
    ):
        config.backend = LocalModelBackend.OLLAMA
        if not config.base_url:
            config.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        super().__init__(config, personality)

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> LocalGenerationResult:
        """Generate with Ollama."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        # Build system prompt with personality
        full_system = self._build_system_prompt(system)

        # Request with logprobs
        request_data = {
            "model": self.config.model_name,
            "prompt": prompt,
            "system": full_system,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }

        # Add logprobs if supported (Ollama 0.1.29+)
        if self.config.logprobs:
            request_data["options"]["logprobs"] = True
            request_data["options"]["top_logprobs"] = self.config.top_logprobs

        start_time = time.time()

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.config.base_url}/api/generate",
                json=request_data,
            )
            response.raise_for_status()
            data = response.json()

        generation_time = time.time() - start_time

        # Parse logprobs if present
        logprobs = None
        if "logprobs" in data:
            logprobs = self._parse_ollama_logprobs(data)

        result = LocalGenerationResult(
            text=data.get("response", ""),
            logprobs=logprobs,
            tokens_generated=data.get("eval_count", 0),
            generation_time=generation_time,
            model=self.config.model_name,
        )

        # Analyze personality if we have logprobs
        if logprobs and self._analyzer:
            result.personality_analysis = self._analyzer.analyze_response(logprobs)

        # Calculate tokens per second
        if result.tokens_generated > 0:
            result.tokens_per_second = result.tokens_generated / generation_time

        return result

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generation from Ollama."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        full_system = self._build_system_prompt(system)

        request_data = {
            "model": self.config.model_name,
            "prompt": prompt,
            "system": full_system,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.config.base_url}/api/generate",
                json=request_data,
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]

    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from Ollama."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        embeddings = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                response = await client.post(
                    f"{self.config.base_url}/api/embeddings",
                    json={
                        "model": self.config.model_name,
                        "prompt": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data.get("embedding", []))

        return np.array(embeddings)

    def _build_system_prompt(self, custom_system: Optional[str] = None) -> str:
        """Build full system prompt with personality."""
        parts = []

        if self.personality:
            parts.append(self.personality.generate_system_prompt())

        if custom_system:
            parts.append(custom_system)

        return "\n\n".join(parts)

    def _parse_ollama_logprobs(self, data: dict) -> ResponseLogprobs:
        """Parse Ollama logprobs format."""
        tokens = []

        # Ollama returns tokens and logprobs as parallel arrays
        token_list = data.get("tokens", [])
        logprob_list = data.get("logprobs", [])
        top_logprobs_list = data.get("top_logprobs", [])

        for i, (token, logprob) in enumerate(zip(token_list, logprob_list)):
            top_lps = {}
            if i < len(top_logprobs_list):
                for alt in top_logprobs_list[i]:
                    top_lps[alt.get("token", "")] = alt.get("logprob", 0.0)

            tokens.append(TokenLogprob(
                token=token,
                logprob=logprob,
                top_logprobs=top_lps,
                position=i,
            ))

        return ResponseLogprobs(
            tokens=tokens,
            model=self.config.model_name,
        )


# ─────────────────────────────────────────────────────────────────────────────
# vLLM Client
# ─────────────────────────────────────────────────────────────────────────────


class VLLMClient(LocalModelClient):
    """Client for vLLM local inference server."""

    def __init__(
        self,
        config: LocalModelConfig,
        personality: Optional[Personality] = None,
    ):
        config.backend = LocalModelBackend.VLLM
        if not config.base_url:
            config.base_url = "http://localhost:8000"
        super().__init__(config, personality)

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> LocalGenerationResult:
        """Generate with vLLM OpenAI-compatible endpoint."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        full_system = self._build_system_prompt(system)

        # Build messages for chat completions
        messages = []
        if full_system:
            messages.append({"role": "system", "content": full_system})
        messages.append({"role": "user", "content": prompt})

        # Get steering biases
        logit_bias = {}
        if self._steering and self.config.use_steering_biases:
            # vLLM accepts token IDs, would need tokenizer to convert
            # For now, skip direct logit bias
            pass

        request_data = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "logprobs": self.config.logprobs,
            "top_logprobs": self.config.top_logprobs if self.config.logprobs else None,
        }

        start_time = time.time()

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.config.base_url}/v1/chat/completions",
                json=request_data,
                headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
            )
            response.raise_for_status()
            data = response.json()

        generation_time = time.time() - start_time

        # Extract response
        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")

        # Parse logprobs
        logprobs = None
        if "logprobs" in choice and choice["logprobs"]:
            logprobs = parse_openai_logprobs(choice["logprobs"])

        result = LocalGenerationResult(
            text=text,
            logprobs=logprobs,
            tokens_generated=data.get("usage", {}).get("completion_tokens", 0),
            generation_time=generation_time,
            model=self.config.model_name,
        )

        if logprobs and self._analyzer:
            result.personality_analysis = self._analyzer.analyze_response(logprobs)

        return result

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream from vLLM."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        full_system = self._build_system_prompt(system)
        messages = []
        if full_system:
            messages.append({"role": "system", "content": full_system})
        messages.append({"role": "user", "content": prompt})

        request_data = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.config.base_url}/v1/chat/completions",
                json=request_data,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]

    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from vLLM."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.config.base_url}/v1/embeddings",
                json={
                    "model": self.config.model_name,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()

        embeddings = [item["embedding"] for item in data.get("data", [])]
        return np.array(embeddings)

    def _build_system_prompt(self, custom_system: Optional[str] = None) -> str:
        parts = []
        if self.personality:
            parts.append(self.personality.generate_system_prompt())
        if custom_system:
            parts.append(custom_system)
        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# llama.cpp Client (via llama-cpp-python)
# ─────────────────────────────────────────────────────────────────────────────


class LlamaCppClient(LocalModelClient):
    """Client for llama.cpp via llama-cpp-python."""

    def __init__(
        self,
        config: LocalModelConfig,
        personality: Optional[Personality] = None,
        model_path: Optional[str] = None,
    ):
        config.backend = LocalModelBackend.LLAMACPP
        super().__init__(config, personality)
        self.model_path = model_path or config.model_name
        self._llm = None

    def _get_llm(self):
        """Lazy load the model."""
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError("llama-cpp-python required: pip install llama-cpp-python")

            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.num_gpu_layers,
                logits_all=self.config.logprobs,  # Enable logprobs
                verbose=False,
            )
        return self._llm

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> LocalGenerationResult:
        """Generate with llama.cpp."""
        llm = self._get_llm()

        full_system = self._build_system_prompt(system)

        # Format prompt for chat
        full_prompt = self._format_chat_prompt(full_system, prompt)

        # Get logit bias from personality steering
        logit_bias = None
        if self._steering and self.config.use_steering_biases:
            # Would need to convert tokens to IDs
            # llama.cpp expects {token_id: bias}
            pass

        start_time = time.time()

        # Run in executor since llama.cpp is sync
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            lambda: llm(
                full_prompt,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                top_k=kwargs.get("top_k", self.config.top_k),
                logprobs=self.config.top_logprobs if self.config.logprobs else None,
                echo=False,
            )
        )

        generation_time = time.time() - start_time

        # Extract text
        text = output.get("choices", [{}])[0].get("text", "")

        # Parse logprobs
        logprobs = None
        if self.config.logprobs and "choices" in output:
            choice_logprobs = output["choices"][0].get("logprobs")
            if choice_logprobs:
                logprobs = self._parse_llamacpp_logprobs(choice_logprobs)

        result = LocalGenerationResult(
            text=text,
            logprobs=logprobs,
            tokens_generated=output.get("usage", {}).get("completion_tokens", 0),
            generation_time=generation_time,
            model=self.config.model_name,
        )

        if logprobs and self._analyzer:
            result.personality_analysis = self._analyzer.analyze_response(logprobs)

        return result

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream from llama.cpp."""
        llm = self._get_llm()

        full_system = self._build_system_prompt(system)
        full_prompt = self._format_chat_prompt(full_system, prompt)

        # llama.cpp streaming is sync, wrap in async
        loop = asyncio.get_event_loop()

        def stream_gen():
            for output in llm(
                full_prompt,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                stream=True,
            ):
                yield output.get("choices", [{}])[0].get("text", "")

        for text in stream_gen():
            yield text
            await asyncio.sleep(0)  # Yield control

    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings (if model supports it)."""
        llm = self._get_llm()

        loop = asyncio.get_event_loop()

        embeddings = []
        for text in texts:
            emb = await loop.run_in_executor(
                None,
                lambda t=text: llm.embed(t)
            )
            embeddings.append(emb)

        return np.array(embeddings)

    def _build_system_prompt(self, custom_system: Optional[str] = None) -> str:
        parts = []
        if self.personality:
            parts.append(self.personality.generate_system_prompt())
        if custom_system:
            parts.append(custom_system)
        return "\n\n".join(parts)

    def _format_chat_prompt(self, system: str, user: str) -> str:
        """Format as chat prompt (Llama-2 style by default)."""
        if system:
            return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
        return f"[INST] {user} [/INST]"

    def _parse_llamacpp_logprobs(self, logprobs_data: dict) -> ResponseLogprobs:
        """Parse llama.cpp logprobs format."""
        tokens = []

        token_list = logprobs_data.get("tokens", [])
        logprob_list = logprobs_data.get("token_logprobs", [])
        top_logprobs_list = logprobs_data.get("top_logprobs", [])

        for i, (token, logprob) in enumerate(zip(token_list, logprob_list)):
            if logprob is None:
                continue

            top_lps = {}
            if i < len(top_logprobs_list) and top_logprobs_list[i]:
                top_lps = top_logprobs_list[i]

            tokens.append(TokenLogprob(
                token=token,
                logprob=logprob,
                top_logprobs=top_lps,
                position=i,
            ))

        return ResponseLogprobs(tokens=tokens, model=self.config.model_name)


# ─────────────────────────────────────────────────────────────────────────────
# MLX-LM Client (Apple Silicon Optimized) - PRIORITY
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MLXModelConfig:
    """Extended configuration for MLX models."""
    model_path: str  # HuggingFace model path or local path

    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    repetition_penalty: float = 1.1
    repetition_context_size: int = 20

    # MLX-specific
    quantize: bool = False  # Use 4-bit quantization
    trust_remote_code: bool = True

    # Vision settings (for mlx-vlm)
    max_pixels: Tuple[int, int] = (224, 224)
    fps: float = 1.0  # For video

    # Personality
    personality_strength: float = 1.0


class MLXClient(LocalModelClient):
    """Client for MLX-LM and MLX-VLM on Apple Silicon.

    Supports:
    - Text generation with mlx-lm
    - Vision-language models with mlx-vlm
    - Audio/video understanding
    - Personality-aware generation

    Usage:
        # Option 1: Pass MLXModelConfig directly
        config = MLXModelConfig(model_path="mlx-community/Llama-3.2-3B-Instruct-4bit")
        client = MLXClient(config, personality=personality)

        # Option 2: Pass LocalModelConfig with mlx_config
        base_config = LocalModelConfig(backend=LocalModelBackend.MLXLM, model_name="model")
        mlx_config = MLXModelConfig(model_path="mlx-community/Llama-3.2-3B-Instruct-4bit")
        client = MLXClient(base_config, personality=personality, mlx_config=mlx_config)
    """

    def __init__(
        self,
        config: Union[LocalModelConfig, MLXModelConfig],
        personality: Optional[Personality] = None,
        mlx_config: Optional[MLXModelConfig] = None,
    ):
        # Handle MLXModelConfig passed directly as config
        if isinstance(config, MLXModelConfig):
            self.mlx_config = config
            # Create a LocalModelConfig for the base class
            base_config = LocalModelConfig(
                backend=LocalModelBackend.MLXLM,
                model_name=config.model_path,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )
            super().__init__(base_config, personality)
        else:
            config.backend = LocalModelBackend.MLXLM
            super().__init__(config, personality)
            self.mlx_config = mlx_config or MLXModelConfig(model_path=config.model_name)
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._model_config = None
        self._is_vlm = False  # Vision-language model flag

    def _load_model(self):
        """Lazy load the MLX model."""
        if self._model is not None:
            return

        model_path = self.mlx_config.model_path

        # Try mlx-vlm first (for vision models)
        try:
            from mlx_vlm import load as vlm_load
            from mlx_vlm.utils import load_config as vlm_load_config

            self._model, self._processor = vlm_load(model_path)
            self._model_config = vlm_load_config(model_path)
            self._is_vlm = True
            logger.info(f"Loaded MLX-VLM model: {model_path}")
            return
        except Exception as e:
            logger.debug(f"Not a VLM model, trying mlx-lm: {e}")

        # Fall back to mlx-lm for text-only models
        try:
            from mlx_lm import load as lm_load

            self._model, self._tokenizer = lm_load(model_path)
            self._is_vlm = False
            logger.info(f"Loaded MLX-LM model: {model_path}")
        except ImportError:
            raise ImportError(
                "MLX-LM required. Install with: pip install mlx-lm mlx-vlm"
            )

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        video: Optional[str] = None,
        **kwargs,
    ) -> LocalGenerationResult:
        """Generate with MLX model.

        Args:
            prompt: Text prompt
            system: System prompt (combined with personality)
            images: List of image paths/URLs for VLM
            audio: List of audio file paths for multimodal
            video: Video file path for video understanding
            **kwargs: Additional generation parameters
        """
        self._load_model()

        full_system = self._build_system_prompt(system)
        start_time = time.time()

        loop = asyncio.get_event_loop()

        if self._is_vlm:
            output = await loop.run_in_executor(
                None,
                lambda: self._generate_vlm(prompt, full_system, images, audio, video, **kwargs)
            )
        else:
            output = await loop.run_in_executor(
                None,
                lambda: self._generate_lm(prompt, full_system, **kwargs)
            )

        generation_time = time.time() - start_time

        result = LocalGenerationResult(
            text=output.get("text", ""),
            tokens_generated=output.get("tokens", 0),
            generation_time=generation_time,
            model=self.mlx_config.model_path,
            tokens_per_second=output.get("tokens", 0) / generation_time if generation_time > 0 else 0,
        )

        # Parse logprobs if available
        if "logprobs" in output and output["logprobs"]:
            result.logprobs = self._parse_mlx_logprobs(output["logprobs"])
            if self._analyzer:
                result.personality_analysis = self._analyzer.analyze_response(result.logprobs)

        return result

    def _generate_vlm(
        self,
        prompt: str,
        system: str,
        images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        video: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Generate with MLX-VLM."""
        from mlx_vlm import generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # Prepare inputs
        image_list = images or []
        audio_list = audio or []

        # Build formatted prompt
        formatted_prompt = apply_chat_template(
            self._processor,
            self._model_config,
            prompt if not system else f"{system}\n\n{prompt}",
            num_images=len(image_list),
            num_audios=len(audio_list) if audio_list else 0,
        )

        # Generate
        output = vlm_generate(
            self._model,
            self._processor,
            formatted_prompt,
            image_list if image_list else None,
            audio=audio_list if audio_list else None,
            max_tokens=kwargs.get("max_tokens", self.mlx_config.max_tokens),
            temperature=kwargs.get("temperature", self.mlx_config.temperature),
            top_p=kwargs.get("top_p", self.mlx_config.top_p),
            verbose=False,
        )

        return {"text": output, "tokens": len(output.split())}  # Approximate

    def _generate_lm(self, prompt: str, system: str, **kwargs) -> dict:
        """Generate with MLX-LM."""
        from mlx_lm import generate as lm_generate
        from mlx_lm.sample_utils import make_sampler

        # Build messages for chat template
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Use tokenizer's chat template if available (handles Qwen, Llama, etc.)
        if hasattr(self._tokenizer, 'apply_chat_template'):
            try:
                full_prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # Disable Qwen3 thinking mode
                )
            except TypeError:
                # Fallback if enable_thinking not supported
                full_prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            # Fallback to simple format
            if system:
                full_prompt = f"System: {system}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\n\nAssistant:"

        # Create sampler with temperature and top_p
        temp = kwargs.get("temperature", self.mlx_config.temperature)
        top_p = kwargs.get("top_p", self.mlx_config.top_p)
        sampler = make_sampler(temp=temp, top_p=top_p)

        output = lm_generate(
            self._model,
            self._tokenizer,
            prompt=full_prompt,
            max_tokens=kwargs.get("max_tokens", self.mlx_config.max_tokens),
            sampler=sampler,
            verbose=False,
        )

        return {"text": output, "tokens": len(output.split())}

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generation from MLX model."""
        self._load_model()

        full_system = self._build_system_prompt(system)

        if self._is_vlm:
            # VLM streaming
            from mlx_vlm import stream_generate
            from mlx_vlm.prompt_utils import apply_chat_template

            image_list = images or []
            formatted_prompt = apply_chat_template(
                self._processor,
                self._model_config,
                prompt if not full_system else f"{full_system}\n\n{prompt}",
                num_images=len(image_list),
            )

            for token in stream_generate(
                self._model,
                self._processor,
                formatted_prompt,
                image_list if image_list else None,
                max_tokens=kwargs.get("max_tokens", self.mlx_config.max_tokens),
                temperature=kwargs.get("temperature", self.mlx_config.temperature),
            ):
                yield token
                await asyncio.sleep(0)
        else:
            # Text-only streaming
            from mlx_lm import stream_generate as lm_stream
            from mlx_lm.sample_utils import make_sampler

            if full_system:
                full_prompt = f"<|system|>\n{full_system}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            else:
                full_prompt = prompt

            # Create sampler for streaming
            temp = kwargs.get("temperature", self.mlx_config.temperature)
            top_p = kwargs.get("top_p", self.mlx_config.top_p)
            sampler = make_sampler(temp=temp, top_p=top_p)

            for response in lm_stream(
                self._model,
                self._tokenizer,
                prompt=full_prompt,
                max_tokens=kwargs.get("max_tokens", self.mlx_config.max_tokens),
                sampler=sampler,
            ):
                yield response.text
                await asyncio.sleep(0)

    async def generate_with_image(
        self,
        prompt: str,
        image_paths: List[str],
        system: Optional[str] = None,
        **kwargs,
    ) -> LocalGenerationResult:
        """Convenience method for image-based generation."""
        return await self.generate(prompt, system, images=image_paths, **kwargs)

    async def generate_with_audio(
        self,
        prompt: str,
        audio_paths: List[str],
        system: Optional[str] = None,
        **kwargs,
    ) -> LocalGenerationResult:
        """Convenience method for audio-based generation."""
        return await self.generate(prompt, system, audio=audio_paths, **kwargs)

    async def generate_with_video(
        self,
        prompt: str,
        video_path: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> LocalGenerationResult:
        """Convenience method for video understanding."""
        return await self.generate(prompt, system, video=video_path, **kwargs)

    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using MLX model."""
        self._load_model()

        # MLX doesn't have built-in embeddings, use mean pooling of hidden states
        # For now, return placeholder - would need custom implementation
        logger.warning("MLX embeddings not fully implemented, returning zeros")
        return np.zeros((len(texts), 768))

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[str]] = None,
        **kwargs,
    ) -> LocalGenerationResult:
        """Chat-style interaction with message history.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            images: Optional images for VLM
        """
        # Extract system message and build conversation
        system = None
        conversation = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                conversation.append(msg)

        # Build prompt from conversation
        prompt_parts = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts)
        if conversation and conversation[-1]["role"] == "user":
            prompt += "\nAssistant:"

        return await self.generate(prompt, system, images=images, **kwargs)

    def _build_system_prompt(self, custom_system: Optional[str] = None) -> str:
        """Build full system prompt with personality."""
        parts = []
        if self.personality:
            parts.append(self.personality.generate_system_prompt())
        if custom_system:
            parts.append(custom_system)
        return "\n\n".join(parts)

    def _parse_mlx_logprobs(self, logprobs_data: Any) -> ResponseLogprobs:
        """Parse MLX logprobs format."""
        tokens = []

        if isinstance(logprobs_data, list):
            for i, item in enumerate(logprobs_data):
                if isinstance(item, dict):
                    tokens.append(TokenLogprob(
                        token=item.get("token", ""),
                        logprob=item.get("logprob", 0.0),
                        top_logprobs=item.get("top_logprobs", {}),
                        position=i,
                    ))
                elif isinstance(item, tuple) and len(item) >= 2:
                    tokens.append(TokenLogprob(
                        token=str(item[0]),
                        logprob=float(item[1]),
                        position=i,
                    ))

        return ResponseLogprobs(tokens=tokens, model=self.mlx_config.model_path)

    @property
    def is_vision_model(self) -> bool:
        """Check if this is a vision-language model."""
        self._load_model()
        return self._is_vlm

    def unload(self):
        """Unload model from memory."""
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._model_config = None
        logger.info(f"Unloaded MLX model: {self.mlx_config.model_path}")


class MLXServerClient(LocalModelClient):
    """Client for MLX-VLM server (FastAPI endpoint)."""

    def __init__(
        self,
        config: LocalModelConfig,
        personality: Optional[Personality] = None,
        server_url: str = "http://localhost:8080",
    ):
        config.backend = LocalModelBackend.MLXLM
        if not config.base_url:
            config.base_url = server_url
        super().__init__(config, personality)

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        **kwargs,
    ) -> LocalGenerationResult:
        """Generate via MLX-VLM server."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        full_system = self._build_system_prompt(system)

        # Build messages in OpenAI format
        messages = []
        if full_system:
            messages.append({"role": "system", "content": full_system})

        # Build user message content
        content = []
        content.append({"type": "text", "text": prompt})

        if images:
            for img in images:
                content.append({"type": "input_image", "image_url": img})

        if audio:
            for aud in audio:
                content.append({"type": "input_audio", "input_audio": aud})

        messages.append({"role": "user", "content": content if len(content) > 1 else prompt})

        request_data = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": False,
        }

        start_time = time.time()

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.config.base_url}/chat/completions",
                json=request_data,
            )
            response.raise_for_status()
            data = response.json()

        generation_time = time.time() - start_time

        # Extract response
        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")

        result = LocalGenerationResult(
            text=text,
            tokens_generated=data.get("usage", {}).get("completion_tokens", 0),
            generation_time=generation_time,
            model=self.config.model_name,
        )

        return result

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream from MLX-VLM server."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        full_system = self._build_system_prompt(system)

        messages = []
        if full_system:
            messages.append({"role": "system", "content": full_system})

        content = [{"type": "text", "text": prompt}]
        if images:
            for img in images:
                content.append({"type": "input_image", "image_url": img})

        messages.append({"role": "user", "content": content if len(content) > 1 else prompt})

        request_data = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.config.base_url}/chat/completions",
                json=request_data,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]

    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """MLX server doesn't support embeddings yet."""
        logger.warning("MLX server embeddings not supported")
        return np.zeros((len(texts), 768))

    async def list_models(self) -> List[str]:
        """List available models on server."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.config.base_url}/models")
            response.raise_for_status()
            data = response.json()

        return data.get("models", [])

    async def unload_model(self):
        """Unload current model from server."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self.config.base_url}/unload")
            response.raise_for_status()

    def _build_system_prompt(self, custom_system: Optional[str] = None) -> str:
        parts = []
        if self.personality:
            parts.append(self.personality.generate_system_prompt())
        if custom_system:
            parts.append(custom_system)
        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Personality Embeddings
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PersonalityEmbedding:
    """Embedding representation of a personality."""
    personality_name: str
    archetype: str
    embedding: np.ndarray
    trait_vector: np.ndarray  # Direct trait values as vector
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PersonalityEmbedder:
    """Generate and compare personality embeddings."""

    def __init__(self, client: LocalModelClient):
        self.client = client
        self._cache: Dict[str, PersonalityEmbedding] = {}

    async def embed_personality(self, personality: Personality) -> PersonalityEmbedding:
        """Create embedding for a personality."""
        # Check cache
        cache_key = f"{personality.name}_{personality.crystallization_level}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate trait vector (direct numeric representation)
        trait_vector = self._personality_to_vector(personality)

        # Generate semantic embedding from personality description
        description = self._personality_to_description(personality)
        embeddings = await self.client.get_embeddings([description])
        embedding = embeddings[0] if len(embeddings) > 0 else np.zeros(768)

        result = PersonalityEmbedding(
            personality_name=personality.name,
            archetype=personality.archetype,
            embedding=embedding,
            trait_vector=trait_vector,
        )

        self._cache[cache_key] = result
        return result

    async def find_similar(
        self,
        personality: Personality,
        candidates: List[Personality],
        top_k: int = 5,
        use_semantic: bool = True,
    ) -> List[Tuple[Personality, float]]:
        """Find similar personalities."""
        target_emb = await self.embed_personality(personality)

        similarities = []
        for candidate in candidates:
            cand_emb = await self.embed_personality(candidate)

            if use_semantic:
                # Cosine similarity of semantic embeddings
                sim = self._cosine_similarity(target_emb.embedding, cand_emb.embedding)
            else:
                # Euclidean distance of trait vectors
                dist = np.linalg.norm(target_emb.trait_vector - cand_emb.trait_vector)
                sim = 1.0 / (1.0 + dist)

            similarities.append((candidate, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def interpolate_personalities(
        self,
        personality_a: Personality,
        personality_b: Personality,
        alpha: float = 0.5,
    ) -> Personality:
        """Create interpolated personality between two."""
        # Interpolate trait vectors
        vec_a = self._personality_to_vector(personality_a)
        vec_b = self._personality_to_vector(personality_b)

        interpolated = vec_a * (1 - alpha) + vec_b * alpha

        # Create new personality from interpolated vector
        new_personality = Personality(
            name=f"{personality_a.name}_{personality_b.name}_blend_{alpha:.2f}",
            archetype=f"Blend of {personality_a.archetype} and {personality_b.archetype}",
        )

        # Set traits from interpolated vector
        self._vector_to_personality(interpolated, new_personality)

        return new_personality

    def _personality_to_vector(self, personality: Personality) -> np.ndarray:
        """Convert personality to numeric vector."""
        values = []

        # Trait dimensions
        for trait in TraitDimension:
            values.append(personality.traits.get_trait(trait))

        # Voice pattern values
        values.extend([
            personality.voice.vocabulary_level,
            personality.voice.sentence_complexity,
            personality.voice.use_analogies,
            personality.voice.use_examples,
            personality.voice.use_hedging,
            personality.voice.use_emphasis,
        ])

        # Behavioral values
        values.extend([
            personality.behavior.risk_tolerance,
            personality.behavior.initiative_level,
            personality.behavior.question_asking,
            personality.behavior.self_correction,
        ])

        return np.array(values)

    def _vector_to_personality(self, vector: np.ndarray, personality: Personality):
        """Set personality values from vector."""
        idx = 0

        # Trait dimensions
        for trait in TraitDimension:
            personality.traits.set_trait(trait, float(vector[idx]))
            idx += 1

        # Voice pattern
        personality.voice.vocabulary_level = float(vector[idx]); idx += 1
        personality.voice.sentence_complexity = float(vector[idx]); idx += 1
        personality.voice.use_analogies = float(vector[idx]); idx += 1
        personality.voice.use_examples = float(vector[idx]); idx += 1
        personality.voice.use_hedging = float(vector[idx]); idx += 1
        personality.voice.use_emphasis = float(vector[idx]); idx += 1

        # Behavioral
        personality.behavior.risk_tolerance = float(vector[idx]); idx += 1
        personality.behavior.initiative_level = float(vector[idx]); idx += 1
        personality.behavior.question_asking = float(vector[idx]); idx += 1
        personality.behavior.self_correction = float(vector[idx]); idx += 1

    def _personality_to_description(self, personality: Personality) -> str:
        """Generate text description for embedding."""
        return personality.generate_system_prompt()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# ─────────────────────────────────────────────────────────────────────────────
# Model-Specific Personality Adaptation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelPersonalityProfile:
    """How well a model expresses different personality traits."""
    model_name: str
    trait_strengths: Dict[TraitDimension, float]  # How well model expresses each trait
    voice_compatibility: Dict[ToneRegister, float]  # How natural each tone is
    optimal_temperature: float = 0.7
    optimal_top_p: float = 0.9
    notes: str = ""


class PersonalityAdapter:
    """Adapt personality prompts for specific models."""

    # Known model profiles
    MODEL_PROFILES: Dict[str, ModelPersonalityProfile] = {
        "llama2": ModelPersonalityProfile(
            model_name="llama2",
            trait_strengths={
                TraitDimension.AGREEABLENESS: 0.8,
                TraitDimension.CONSCIENTIOUSNESS: 0.7,
                TraitDimension.ANALYTICITY: 0.6,
                TraitDimension.CREATIVITY: 0.5,
                TraitDimension.ASSERTIVENESS: 0.4,
            },
            voice_compatibility={
                ToneRegister.PROFESSIONAL: 0.9,
                ToneRegister.FRIENDLY: 0.8,
                ToneRegister.CASUAL: 0.6,
                ToneRegister.PLAYFUL: 0.4,
            },
            optimal_temperature=0.7,
            notes="Strong at helpful, harmless responses. May resist assertive personalities.",
        ),
        "mistral": ModelPersonalityProfile(
            model_name="mistral",
            trait_strengths={
                TraitDimension.ANALYTICITY: 0.85,
                TraitDimension.CREATIVITY: 0.7,
                TraitDimension.ASSERTIVENESS: 0.65,
                TraitDimension.AGREEABLENESS: 0.6,
                TraitDimension.HUMOR: 0.5,
            },
            voice_compatibility={
                ToneRegister.PROFESSIONAL: 0.85,
                ToneRegister.CASUAL: 0.75,
                ToneRegister.PLAYFUL: 0.6,
                ToneRegister.AUTHORITATIVE: 0.7,
            },
            optimal_temperature=0.8,
            notes="More willing to adopt varied personalities. Good for creative tasks.",
        ),
        "codellama": ModelPersonalityProfile(
            model_name="codellama",
            trait_strengths={
                TraitDimension.ANALYTICITY: 0.95,
                TraitDimension.CONSCIENTIOUSNESS: 0.85,
                TraitDimension.FORMALITY: 0.8,
                TraitDimension.CREATIVITY: 0.4,
                TraitDimension.HUMOR: 0.2,
            },
            voice_compatibility={
                ToneRegister.PROFESSIONAL: 0.95,
                ToneRegister.FORMAL: 0.9,
                ToneRegister.CASUAL: 0.3,
            },
            optimal_temperature=0.3,
            notes="Best for technical personalities. Struggles with casual/playful.",
        ),
        "phi": ModelPersonalityProfile(
            model_name="phi",
            trait_strengths={
                TraitDimension.ANALYTICITY: 0.8,
                TraitDimension.PATIENCE: 0.7,
                TraitDimension.CONSCIENTIOUSNESS: 0.75,
            },
            voice_compatibility={
                ToneRegister.PROFESSIONAL: 0.85,
                ToneRegister.FRIENDLY: 0.7,
            },
            optimal_temperature=0.6,
            notes="Small model, best for simple personalities.",
        ),
    }

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.profile = self._get_profile(model_name)

    def _get_profile(self, model_name: str) -> ModelPersonalityProfile:
        """Get or create profile for model."""
        # Check exact match
        if model_name in self.MODEL_PROFILES:
            return self.MODEL_PROFILES[model_name]

        # Check partial match
        model_lower = model_name.lower()
        for key, profile in self.MODEL_PROFILES.items():
            if key in model_lower:
                return profile

        # Return default profile
        return ModelPersonalityProfile(
            model_name=model_name,
            trait_strengths={t: 0.5 for t in TraitDimension},
            voice_compatibility={t: 0.5 for t in ToneRegister},
        )

    def adapt_personality(self, personality: Personality) -> Personality:
        """Adapt personality for this model's strengths."""
        adapted = Personality(
            name=f"{personality.name}_adapted_{self.model_name}",
            archetype=personality.archetype,
            core_motivation=personality.core_motivation,
        )

        # Copy and adjust traits based on model strengths
        for trait in TraitDimension:
            original = personality.traits.get_trait(trait)
            model_strength = self.profile.trait_strengths.get(trait, 0.5)

            # Moderate extreme traits toward model's comfort zone
            if model_strength < 0.4 and original > 0.7:
                # Model weak at this, reduce intensity
                adapted.traits.set_trait(trait, original * 0.8)
            elif model_strength > 0.7 and original < 0.3:
                # Model strong at this, can maintain low value
                adapted.traits.set_trait(trait, original)
            else:
                adapted.traits.set_trait(trait, original)

        # Adjust voice for model compatibility
        tone_compat = self.profile.voice_compatibility.get(
            personality.voice.primary_tone, 0.5
        )

        if tone_compat < 0.4:
            # Find more compatible tone
            best_tone = max(
                self.profile.voice_compatibility.items(),
                key=lambda x: x[1]
            )[0]
            adapted.voice.primary_tone = best_tone
        else:
            adapted.voice.primary_tone = personality.voice.primary_tone

        # Copy other voice settings
        adapted.voice.vocabulary_level = personality.voice.vocabulary_level
        adapted.voice.use_examples = personality.voice.use_examples
        adapted.voice.use_analogies = personality.voice.use_analogies

        return adapted

    def get_optimal_params(self) -> Dict[str, Any]:
        """Get optimal generation parameters for model."""
        return {
            "temperature": self.profile.optimal_temperature,
            "top_p": self.profile.optimal_top_p,
        }

    def estimate_personality_fit(self, personality: Personality) -> float:
        """Estimate how well model can express this personality (0-1)."""
        scores = []

        # Check trait compatibility
        for trait in TraitDimension:
            trait_val = personality.traits.get_trait(trait)
            model_strength = self.profile.trait_strengths.get(trait, 0.5)

            # High trait needs high model strength
            if trait_val > 0.7:
                scores.append(model_strength)
            # Low trait is easier
            elif trait_val < 0.3:
                scores.append(0.8)
            else:
                scores.append(0.6)

        # Check voice compatibility
        tone_score = self.profile.voice_compatibility.get(
            personality.voice.primary_tone, 0.5
        )
        scores.append(tone_score)

        return statistics.mean(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning Data Generation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FineTuningExample:
    """Single fine-tuning example for personality."""
    prompt: str
    response: str
    system: str
    personality_name: str
    trait_tags: List[str] = field(default_factory=list)


class PersonalityFineTuneGenerator:
    """Generate fine-tuning data for personality alignment."""

    def __init__(self, personality: Personality, client: LocalModelClient):
        self.personality = personality
        self.client = client

    async def generate_examples(
        self,
        prompts: List[str],
        num_variations: int = 3,
    ) -> List[FineTuningExample]:
        """Generate fine-tuning examples from prompts."""
        examples = []
        system_prompt = self.personality.generate_system_prompt()

        for prompt in prompts:
            for i in range(num_variations):
                # Generate response with personality
                result = await self.client.generate(
                    prompt=prompt,
                    system=system_prompt,
                    temperature=0.7 + (i * 0.1),  # Vary temperature
                )

                # Analyze if response matches personality
                if result.personality_analysis:
                    alignment = result.personality_analysis.get("alignment", 0)
                    if alignment < 0.5:
                        continue  # Skip poor examples

                # Get dominant traits for tagging
                trait_tags = [
                    t.value for t, v in self.personality.traits.dominant_traits()
                ]

                examples.append(FineTuningExample(
                    prompt=prompt,
                    response=result.text,
                    system=system_prompt,
                    personality_name=self.personality.name,
                    trait_tags=trait_tags,
                ))

        return examples

    def export_jsonl(self, examples: List[FineTuningExample], path: str):
        """Export examples as JSONL for fine-tuning."""
        with open(path, "w") as f:
            for ex in examples:
                record = {
                    "messages": [
                        {"role": "system", "content": ex.system},
                        {"role": "user", "content": ex.prompt},
                        {"role": "assistant", "content": ex.response},
                    ],
                    "personality": ex.personality_name,
                    "traits": ex.trait_tags,
                }
                f.write(json.dumps(record) + "\n")

    def export_alpaca(self, examples: List[FineTuningExample], path: str):
        """Export in Alpaca format."""
        records = []
        for ex in examples:
            records.append({
                "instruction": ex.prompt,
                "input": "",
                "output": ex.response,
                "system": ex.system,
            })

        with open(path, "w") as f:
            json.dump(records, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────


def create_local_client(
    backend: Union[LocalModelBackend, str],
    model_name: str,
    personality: Optional[Personality] = None,
    **kwargs,
) -> LocalModelClient:
    """Create a local model client."""
    if isinstance(backend, str):
        backend = LocalModelBackend(backend)

    config = LocalModelConfig(
        backend=backend,
        model_name=model_name,
        **kwargs,
    )

    if backend == LocalModelBackend.OLLAMA:
        return OllamaClient(config, personality)
    elif backend == LocalModelBackend.VLLM:
        return VLLMClient(config, personality)
    elif backend == LocalModelBackend.LLAMACPP:
        return LlamaCppClient(config, personality, model_path=kwargs.get("model_path"))
    else:
        raise ValueError(f"Unsupported backend: {backend}")


async def analyze_personality_expression(
    client: LocalModelClient,
    personality: Personality,
    test_prompts: List[str],
) -> Dict[str, Any]:
    """Analyze how well a model expresses a personality."""
    client.set_personality(personality)

    results = []
    for prompt in test_prompts:
        result = await client.generate(prompt)
        if result.personality_analysis:
            results.append(result.personality_analysis)

    if not results:
        return {"error": "No analysis results"}

    # Aggregate
    alignments = [r.get("alignment", 0) for r in results]
    authenticities = [r.get("authenticity", 0) for r in results]

    return {
        "mean_alignment": statistics.mean(alignments),
        "mean_authenticity": statistics.mean(authenticities),
        "alignment_std": statistics.stdev(alignments) if len(alignments) > 1 else 0,
        "num_samples": len(results),
        "problem_traits": list(set(
            trait for r in results
            for trait in r.get("problem_traits", [])
        )),
    }
