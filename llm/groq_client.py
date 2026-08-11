"""
Groq API client wrapper.
Used for fast text-only generation (quizzes, flashcards).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, MAX_OUTPUT_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)


def _get_client() -> Groq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
    return Groq(api_key=GROQ_API_KEY)


def generate_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = TEMPERATURE,
    json_mode: bool = False,
) -> str:
    """
    Generate text via Groq (fast inference).

    Args:
        prompt:       User message.
        system_prompt: Optional system prompt.
        max_tokens:   Max tokens in response.
        temperature:  Sampling temperature.
        json_mode:    If True, requests JSON object output (for quiz/flashcard generation).

    Returns:
        Generated text string.
    """
    client = _get_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def generate_json(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict | list:
    """
    Generate a JSON object via Groq and parse it.

    Returns:
        Parsed Python dict or list.
    Raises:
        ValueError if the response cannot be parsed as JSON.
    """
    raw = generate_text(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=0.1,   # low temperature for reliable JSON
        json_mode=True,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Groq returned invalid JSON: %s\n---\n%s", exc, raw)
        raise ValueError(f"Groq response was not valid JSON: {exc}") from exc
