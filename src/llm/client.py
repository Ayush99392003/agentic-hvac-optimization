"""Centralized LLM client integration module.

All calls to external LLM providers (Azure OpenAI, OpenAI, or local OSS models)
pass through this single module as required by system architecture rules.

Supports:
- Azure OpenAI via `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_DEPLOYMENT`
- Cloud OpenAI via `OPENAI_API_KEY`
- Local OSS LLMs (Ollama / vLLM) via `LLM_BASE_URL`
- Dynamic CLI overrides via `set_llm_credentials()`

Architecture reference: Section 3 (LLM Integration — Single Module Rule).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get(
    "AZURE_OPENAI_DEPLOYMENT",
    os.environ.get("LLM_MODEL", "gpt-4o"),
)

# Global configuration store updated via set_llm_credentials
_CONFIG_OVERRIDE_API_KEY: str | None = None
_CONFIG_OVERRIDE_BASE_URL: str | None = None
_CONFIG_OVERRIDE_MODEL: str | None = None


def set_llm_credentials(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> None:
    """Set global LLM credentials override from CLI or caller.

    Args:
        api_key: API key string.
        base_url: Base URL endpoint string.
        model: Model / deployment name string.
    """
    global _CONFIG_OVERRIDE_API_KEY, _CONFIG_OVERRIDE_BASE_URL, _CONFIG_OVERRIDE_MODEL
    if api_key:
        _CONFIG_OVERRIDE_API_KEY = api_key
    if base_url:
        _CONFIG_OVERRIDE_BASE_URL = base_url
    if model:
        _CONFIG_OVERRIDE_MODEL = model


@dataclass
class LLMConfig:
    """Execution configuration parameters for an LLM call."""

    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    max_tokens: int = 1000
    timeout_seconds: float = 30.0


def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    config: LLMConfig | None = None,
) -> str:
    """Send a request to the centralized LLM client.

    Uses AzureOpenAI client if AZURE_OPENAI_ENDPOINT is configured,
    otherwise standard OpenAI / OSS client.

    Args:
        prompt: User / context prompt text.
        system_prompt: Optional system instruction prompt.
        config: Execution configuration.

    Returns:
        String output produced by the LLM.

    Raises:
        RuntimeError: If no API credentials are available or network call fails.
    """
    cfg = config or LLMConfig()
    model_name = _CONFIG_OVERRIDE_MODEL or cfg.model

    # Check for Azure OpenAI environment variables
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_api_key = _CONFIG_OVERRIDE_API_KEY or os.environ.get(
        "AZURE_OPENAI_API_KEY"
    )
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", model_name)
    azure_version = os.environ.get(
        "AZURE_OPENAI_API_VERSION", "2025-04-01-preview"
    )

    api_key = _CONFIG_OVERRIDE_API_KEY or os.environ.get("OPENAI_API_KEY")
    base_url = _CONFIG_OVERRIDE_BASE_URL or os.environ.get("LLM_BASE_URL")

    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError:
        raise RuntimeError(
            "openai package is not installed. Install via `uv add openai`."
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # 1. Azure OpenAI Endpoint
    if azure_endpoint and azure_api_key:
        logger.info(
            "Central LLM Client: invoking AzureOpenAI endpoint %s (deployment=%s)",
            azure_endpoint,
            azure_deployment,
        )
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_version,
            timeout=cfg.timeout_seconds,
        )
        try:
            response = client.chat.completions.create(
                model=azure_deployment,
                messages=messages,
                temperature=cfg.temperature,
                max_completion_tokens=cfg.max_tokens,
            )
        except Exception:
            response = client.chat.completions.create(
                model=azure_deployment,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
        return response.choices[0].message.content or ""

    # 2. Standard Cloud OpenAI or Local OSS Base URL
    if not api_key and not base_url:
        raise RuntimeError(
            "LLM unavailable: OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set. "
            "LLM call cannot proceed."
        )

    logger.info(
        "Central LLM Client: invoking OpenAI endpoint (model=%s)", model_name
    )
    client_kwargs = {"timeout": cfg.timeout_seconds}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=cfg.temperature,
            max_completion_tokens=cfg.max_tokens,
        )
    except Exception:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    return response.choices[0].message.content or ""
