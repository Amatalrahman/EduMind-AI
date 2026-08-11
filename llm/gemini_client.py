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
    GEMINI_API_KEY,
    GEMINI_TEXT_MODEL,
    GEMINI_VISION_MODEL,
    MAX_OUTPUT_TOKENS,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)

# Instantiate a single client reused across calls
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


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
    client = _get_client()
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

    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=contents,
        config=config,
    )
    return response.text


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
    client = _get_client()

    config = _generation_config()
    if system_instruction:
        config.system_instruction = system_instruction

    contents = [
        types.Content(role="user", parts=[
            types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes)),
            types.Part(text=prompt),
        ])
    ]

    response = client.models.generate_content(
        model=GEMINI_VISION_MODEL,
        contents=contents,
        config=config,
    )
    return response.text
