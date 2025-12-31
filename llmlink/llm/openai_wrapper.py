import os
import json
import hashlib
import logging
from pathlib import Path
import numpy as np

from collections.abc import Iterable
import sys

import pipmaster as pm

logger = logging.getLogger("llmlink.openai_wrapper")

from openai import (
    OpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from dataclasses import dataclass

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable
    # concurrent_limit: int = 16

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


import re

def safe_unicode_decode(content):
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match):
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))

    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content.decode("utf-8")
    )

    return decoded_content


import base64
from typing import Any, Optional, Union

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""
    pass


CACHE_DIR = Path(os.path.expanduser("/mnt/nvme1n1p1/Data/openai_inference_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in sorted(obj.items())}
    return repr(obj)


def _canonicalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for message in messages:
        canonical.append(
            {
                "role": message.get("role"),
                "content": message.get("content"),
                "name": message.get("name"),
            }
        )
    return canonical


def _make_cache_key(
    model: str,
    system_prompt: Optional[str],
    messages: list[dict[str, Any]],
    params: dict[str, Any],
) -> str:
    payload = {
        "model": model,
        "system_prompt": system_prompt,
        "messages": _canonicalize_messages(messages),
        "params": _jsonify(params),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_cache(key: str) -> Optional[dict[str, Any]]:
    cache_file = CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            f"Failed to read cache for key {key}: {exc}"
        )
        return None


def _write_cache(key: str, payload: dict[str, Any]) -> None:
    cache_file = CACHE_DIR / f"{key}.json"
    try:
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            f"Failed to write cache for key {key}: {exc}"
        )


def create_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    client_configs: dict[str, Any] | None = None,
) -> OpenAI:
    """Create an OpenAI client with the given configuration.

    Args:
        api_key: OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        base_url: Base URL for the OpenAI API. If None, uses the default OpenAI API URL.
        client_configs: Additional configuration options for the OpenAI client.
            These will override any default configurations but will be overridden by
            explicit parameters (api_key, base_url).

    Returns:
        An OpenAI client instance.
    """
    if not api_key:
        api_key = os.environ["OPENAI_API_KEY"]

    default_headers = {
        "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LlmLink/{os.environ.get('LLMLINK_VERSION', 'dev')}",
        "Content-Type": "application/json",
        "X-Enable-GPT-5-Codex-Preview": "true",
    }

    if client_configs is None:
        client_configs = {}

    merged_configs = {
        **client_configs,
        "default_headers": default_headers,
        "api_key": api_key,
    }

    if base_url is not None:
        merged_configs["base_url"] = base_url
    else:
        merged_configs["base_url"] = os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        )

    return OpenAI(**merged_configs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InvalidResponseError)
    ),
)
def openai_complete_if_cache(
    model: str,
    prompt: str,
    *,
    client: OpenAI,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    token_tracker: Any | None = None,
    response_format: Any | None = None,
    request_params: dict[str, Any] | None = None,
    is_stream: bool = False,
    messages: list[dict[str, Any]] | None = None,
) -> Union[str, Iterable[str]]:
    """Complete a prompt using OpenAI's API with caching and optional COT support."""
    if history_messages is None:
        history_messages = []

    logging.getLogger("openai").setLevel(logging.INFO)

    if messages is None:
        compiled_messages: list[dict[str, Any]] = []
        if system_prompt:
            compiled_messages.append({"role": "system", "content": system_prompt})
        compiled_messages.extend(history_messages)
        compiled_messages.append({"role": "user", "content": prompt})
    else:
        compiled_messages = list(messages)

    logger.debug("===== Entering func of LLM =====")
    logger.debug(f"Model: {model}")
    logger.debug(f"Client: {client}")
    logger.debug(f"Additional request params: {request_params}")
    logger.debug(f"Num of history messages: {len(history_messages)}")
    logger.debug(f"System prompt: {system_prompt}")
    logger.debug(f"Query: {prompt}")
    logger.debug("===== Sending Query to LLM =====")

    request_kwargs = dict(request_params or {})
    if response_format is not None:
        request_kwargs["response_format"] = response_format

    if is_stream and response_format is not None:
        raise ValueError(
            "Streaming is not supported when response_format parsing is requested."
        )

    cache_key: Optional[str] = None
    if not is_stream:
        cache_key = _make_cache_key(model, system_prompt, compiled_messages, request_kwargs)
        cached_entry = _read_cache(cache_key)
        if cached_entry and cached_entry.get("content"):
            cached_usage = cached_entry.get("usage") or {}
            if token_tracker and isinstance(cached_usage, dict):
                token_tracker.add_usage(
                    {
                        "prompt_tokens": cached_usage.get("prompt_tokens", 0),
                        "completion_tokens": cached_usage.get("completion_tokens", 0),
                        "total_tokens": cached_usage.get("total_tokens", 0),
                    }
                )
            logger.debug("Cache hit for OpenAI completion request")
            return cached_entry["content"]

    try:
        if response_format is not None:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=compiled_messages,
                **request_kwargs,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=compiled_messages,
                stream=is_stream,
                **request_kwargs,
            )
    except APIConnectionError as exc:
        logger.error(f"OpenAI API Connection Error: {exc}")
        raise
    except RateLimitError as exc:
        logger.error(f"OpenAI API Rate Limit Error: {exc}")
        raise
    except APITimeoutError as exc:
        logger.error(f"OpenAI API Timeout Error: {exc}")
        raise
    except Exception as exc:
        logger.error(
            f"OpenAI API Call Failed,\nModel: {model},\nParams: {request_kwargs}, Got: {exc}"
        )
        raise

    if is_stream:

        def stream_generator() -> Iterable[str]:
            final_chunk_usage = None
            cot_active = False
            cot_started = False
            initial_content_seen = False

            try:
                for chunk in response:
                    if hasattr(chunk, "usage") and chunk.usage:
                        final_chunk_usage = chunk.usage
                        logger.debug(
                            f"Received usage info in streaming chunk: {chunk.usage}"
                        )

                    if not hasattr(chunk, "choices") or not chunk.choices:
                        logger.warning(f"Received chunk without choices: {chunk}")
                        continue

                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta is None:
                        continue

                    content = getattr(delta, "content", None)
                    # Prefer model-specific "reasoning" field for Qwen Thinking models,
                    # fall back to generic "reasoning_content" otherwise.
                    if "Qwen" in model and 'Thinking' in model:
                        reasoning_content = getattr(delta, "reasoning", None) or getattr(
                            delta, "reasoning_content", None
                        )
                    else:
                        reasoning_content = getattr(delta, "reasoning_content", None) or getattr(
                            delta, "reasoning", None
                        )

                    if enable_cot:
                        if content:
                            if not initial_content_seen:
                                initial_content_seen = True
                                if reasoning_content:
                                    cot_active = False
                                    cot_started = False

                            if cot_active:
                                yield "</think>"
                                cot_active = False

                            if r"\u" in content:
                                content = safe_unicode_decode(content.encode("utf-8"))
                            yield content

                        elif reasoning_content:
                            if not initial_content_seen and not cot_started:
                                if not cot_active:
                                    yield "<think>"
                                    cot_active = True
                                    cot_started = True

                            if cot_active:
                                if r"\u" in reasoning_content:
                                    reasoning_content = safe_unicode_decode(
                                        reasoning_content.encode("utf-8")
                                    )
                                yield reasoning_content
                    else:
                        if content:
                            if r"\u" in content:
                                content = safe_unicode_decode(content.encode("utf-8"))
                            yield content

                if enable_cot and cot_active:
                    yield "</think>"
                    cot_active = False

                if token_tracker and final_chunk_usage:
                    token_counts = {
                        "prompt_tokens": getattr(final_chunk_usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            final_chunk_usage, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(final_chunk_usage, "total_tokens", 0),
                    }
                    token_tracker.add_usage(token_counts)
                elif token_tracker:
                    logger.debug("No usage information available in streaming response")
            except Exception as exc:
                if enable_cot and cot_active:
                    try:
                        yield "</think>"
                        cot_active = False
                    except Exception as close_error:
                        logger.warning(
                            f"Failed to close COT tag during exception handling: {close_error}"
                        )
                logger.error(f"Error in stream response: {exc}")
                raise

        return stream_generator()

    if (
        not response
        or not getattr(response, "choices", None)
        or not hasattr(response.choices[0], "message")
    ):
        logger.error("Invalid response from OpenAI API")
        raise InvalidResponseError("Invalid response from OpenAI API")

    message = response.choices[0].message
    content = getattr(message, "content", None)
    # Prefer model-specific "reasoning" field for Qwen Thinking models,
    # fall back to generic "reasoning_content" otherwise.
    if "Qwen" in model and 'Thinking' in model:
        reasoning_content = getattr(message, "reasoning", None) or getattr(
            message, "reasoning_content", None
        )
    else:
        reasoning_content = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )

    # Build final_content. For Qwen Thinking models treat reasoning as the COT content
    final_content = content or ""

    if enable_cot and reasoning_content and str(reasoning_content).strip():
        # decode escaped unicode sequences if present
        if r"\u" in reasoning_content:
            try:
                reasoning_content = safe_unicode_decode(reasoning_content.encode("utf-8"))
            except Exception:
                pass

        # For the Qwen Thinking model, always include reasoning as COT before the visible content.
        if 'Qwen' in model and 'Thinking' in model:
            final_content = f"<think>{reasoning_content}</think>{final_content}"
        else:
            # For other models keep previous behavior: only include reasoning if content is empty
            if not final_content or str(final_content).strip() == "":
                final_content = f"<think>{reasoning_content}</think>{final_content}"
    else:
        final_content = content or ""

    if not final_content or final_content.strip() == "":
        logger.error("Received empty content from OpenAI API")
        raise InvalidResponseError("Received empty content from OpenAI API")

    if r"\u" in final_content:
        final_content = safe_unicode_decode(final_content.encode("utf-8"))

    token_counts: dict[str, int] = {}
    if token_tracker and hasattr(response, "usage"):
        token_counts = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        }
        token_tracker.add_usage(token_counts)

    logger.debug(f"Response content len: {len(final_content)}")
    logger.debug(f"Response: {response}")

    if cache_key:
        _write_cache(
            cache_key,
            {
                "content": final_content,
                "usage": token_counts,
                "model": model,
            },
        )

    return final_content


def openai_complete(
    prompt: str,
    *,
    client: OpenAI,
    model_name: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    response_format: Any | None = None,
    request_params: dict[str, Any] | None = None,
    enable_cot: bool = False,
    token_tracker: Any | None = None,
    is_stream: bool = False,
    messages: list[dict[str, Any]] | None = None,
) -> Union[str, Iterable[str]]:
    if history_messages is None:
        history_messages = []
    return openai_complete_if_cache(
        model_name,
        prompt,
        client=client,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        token_tracker=token_tracker,
        response_format=response_format,
        request_params=request_params,
        is_stream=is_stream,
        messages=messages,
    )


if __name__ == "__main__":
    # Initialize logger
    logger = logging.getLogger("llmlink.openai_wrapper")
    logger.propagate = False  # prevent log message send to root loggger
    # Let the main application configure the handlers
    logger.setLevel(logging.INFO)

    # Set httpx logging level to WARNING
    logging.getLogger("httpx").setLevel(logging.WARNING)

    client = create_openai_client(
        api_key="llmlink",
        base_url="http://localhost:1101/v1",
    )

    # First, check available models
    print("Checking available models...")
    models = client.models.list()
    available_models = [model.id for model in models.data]
    print(f"Available models: {available_models}")

    # <<<<<< Debugging openai_complete function <<<<<<
    response = openai_complete(
        "Hello, how many r in strawberry?",
        client=client,
        model_name='',
        is_stream=True,
    )
    # 这边要注意的是,如果以前用了enable_cot, 那么返回的response里面会有<think>和</think>标签,就算以后改成non-thinking了, 这些标签因为调用缓存的原因还是会有的

    # Obtain text from response (handles both non-stream string and streaming iterable)
    if isinstance(response, Iterable) and not isinstance(response, (str, bytes)):
        parts: list[str] = []
        for chunk in response:
            # print streaming chunk as it arrives (optional)
            print(chunk, end="", flush=True)
            parts.append(chunk)
        print()  # newline after stream
        text = "".join(parts)
    else:
        text = response

    print(f"LLM response: {text}")
