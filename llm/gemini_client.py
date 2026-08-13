"""
Gemini API client wrapper using the current google-genai SDK.
Supports text generation and vision (multi-modal) queries.
"""

from __future__ import annotations

import logging
from typing import Optional

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEYS,
    GEMINI_TEXT_MODEL,
    GEMINI_VISION_MODEL,
    MAX_OUTPUT_TOKENS,
    TEMPERATURE,
)
from llm import QuotaExhaustedError

logger = logging.getLogger(__name__)





def _generation_config(
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = TEMPERATURE,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )


def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    history: Optional[list[dict]] = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Generate a text response from Gemini.

    Args:
        prompt:             The user's current message / prompt.
        system_instruction: Optional system prompt.
        history:            List of prior turns: [{"role": "user"|"model", "parts": ["…"]}]
        max_tokens:         Maximum output tokens.
        temperature:        Sampling temperature.

    Returns:
        Generated text string.
    """
    if not GEMINI_API_KEYS:
        raise ValueError("GEMINI_API_KEYS is empty. Please add keys to your .env file.")

    config = _generation_config(max_tokens, temperature)
    if system_instruction:
        config.system_instruction = system_instruction

    # Build contents list: history + current prompt
    contents: list = []
    if history:
        for turn in history:
            role = turn.get("role", "user")
            parts = turn.get("parts", [""])
            contents.append(types.Content(role=role, parts=[types.Part(text=p) for p in parts]))
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    last_err = None
    for api_key in GEMINI_API_KEYS:
        client = genai.Client(api_key=api_key)
        try:
            response = client.models.generate_content(
                model=GEMINI_TEXT_MODEL,
                contents=contents,
                config=config,
            )
            return response.text
        except Exception as exc:
            err_msg = str(exc).lower()
            if "429" in err_msg or "quota" in err_msg or "exhausted" in err_msg or "rate limit" in err_msg:
                logger.warning(f"Gemini API key {api_key[:10]}... exhausted/failed: {exc}. Trying next...")
                last_err = exc
                continue
            else:
                # Re-raise if it's not a quota/rate limit error (e.g. invalid request)
                raise
                
    raise QuotaExhaustedError("All Gemini API keys have exhausted their quota or failed.") from last_err


def generate_with_image(
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/png",
    system_instruction: Optional[str] = None,
) -> str:
    """
    Generate text from a prompt + an image (vision).

    Args:
        prompt:             Instruction for what to describe / analyse.
        image_bytes:        Raw image bytes.
        mime_type:          MIME type of the image ('image/png', 'image/jpeg', etc.)
        system_instruction: Optional system prompt.

    Returns:
        Generated description text.
    """
    if not GEMINI_API_KEYS:
        raise ValueError("GEMINI_API_KEYS is empty. Please add keys to your .env file.")

    config = _generation_config()
    if system_instruction:
        config.system_instruction = system_instruction

    contents = [
        types.Content(role="user", parts=[
            types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes)),
            types.Part(text=prompt),
        ])
    ]

    last_err = None
    for api_key in GEMINI_API_KEYS:
        client = genai.Client(api_key=api_key)
        try:
            response = client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=contents,
                config=config,
            )
            return response.text
        except Exception as exc:
            err_msg = str(exc).lower()
            if "429" in err_msg or "quota" in err_msg or "exhausted" in err_msg or "rate limit" in err_msg:
                logger.warning(f"Gemini API key {api_key[:10]}... exhausted/failed: {exc}. Trying next...")
                last_err = exc
                continue
            else:
                raise
                
    raise QuotaExhaustedError("All Gemini API keys have exhausted their quota or failed.") from last_err
