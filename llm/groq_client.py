"""
Groq API client wrapper.
Used for fast text-only generation (quizzes, flashcards).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from groq import Groq

from config import GROQ_API_KEYS, GROQ_MODEL, MAX_OUTPUT_TOKENS, TEMPERATURE
from llm import QuotaExhaustedError

logger = logging.getLogger(__name__)


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
    if not GROQ_API_KEYS:
        raise ValueError("GROQ_API_KEYS is empty. Please add keys to your .env file.")

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

    last_err = None
    for api_key in GROQ_API_KEYS:
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as exc:
            err_msg = str(exc).lower()
            if "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg or "exhausted" in err_msg:
                logger.warning(f"Groq API key {api_key[:10]}... failed: {exc}. Trying next...")
                last_err = exc
                continue
            else:
                raise
                
    raise QuotaExhaustedError("All Groq API keys have exhausted their quota or failed.") from last_err


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
