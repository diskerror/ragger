"""
API format definitions for inference endpoints.

OpenAI-compatible is the hardcoded default — no file needed.
All other formats are loaded from JSON files.

Search order for format files:
  1. ~/.ragger/formats/<name>.json       (user override)
  2. <formats_dir>/<name>.json           (from config, default /var/ragger/formats)
  3. <package>/formats/<name>.json       (shipped with Ragger)

Custom formats: drop a .json file in ~/.ragger/formats/ or the configured
formats_dir and reference it by name (without .json): format = myformat
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Hardcoded OpenAI default — the "no file needed" format
# -----------------------------------------------------------------------
OPENAI_FORMAT: Dict[str, Any] = {
    "path": "/chat/completions",
    "auth": "bearer",
    "auth_header": "Authorization",
    "auth_prefix": "Bearer ",
    "auth_extra": {},
    "request_transform": "openai",
    "response_content": "choices[0].message.content",
    "stream_content": "choices[0].delta.content",
    "stream_type_field": None,
    "stream_type_value": None,
    "stream_stop": "[DONE]",
}

# -----------------------------------------------------------------------
# Format search dirs
# -----------------------------------------------------------------------
_PACKAGE_FORMATS_DIR = str(Path(__file__).parent.parent / "formats")

# Configurable system formats dir (set via init_formats_dir)
_system_formats_dir: str = "/var/ragger/formats"


def init_formats_dir(path: str):
    """Set the system formats directory from config. Call once at startup."""
    global _system_formats_dir, _cache
    _system_formats_dir = os.path.expanduser(path)
    _cache.clear()  # reload on next access


def _format_search_dirs() -> List[str]:
    return [
        os.path.expanduser("~/.ragger/formats"),
        _system_formats_dir,
        _PACKAGE_FORMATS_DIR,
    ]

# -----------------------------------------------------------------------
# Cache
# -----------------------------------------------------------------------
_cache: Dict[str, Dict[str, Any]] = {}


def _load_format_file(name: str) -> Optional[Dict[str, Any]]:
    """Find and load a format JSON file by name."""
    for d in _format_search_dirs():
        path = os.path.join(d, f"{name}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    fmt = json.load(f)
                logger.debug(f"Loaded format {name!r} from {path}")
                return fmt
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load format {path}: {e}")
    return None


def get_format(name: str) -> Dict[str, Any]:
    """
    Get a format definition by name.

    All formats are loaded from JSON files. The hardcoded OPENAI_FORMAT
    is only used as a fallback if no openai.json file is found, and as
    the base for merging partial schemas.
    """
    if not name:
        name = "openai"

    if name in _cache:
        return _cache[name]

    fmt = _load_format_file(name)
    if fmt is None:
        if name == "openai":
            # Fallback: use hardcoded default if openai.json not found
            _cache[name] = OPENAI_FORMAT
            return OPENAI_FORMAT
        raise KeyError(
            f"Unknown API format {name!r}. "
            f"No {name}.json found in: {', '.join(_format_search_dirs())}"
        )

    # Merge with OpenAI defaults so partial schemas work
    merged = {**OPENAI_FORMAT, **fmt}
    _cache[name] = merged
    return merged


def list_formats() -> List[str]:
    """List available format names (openai + any .json files found)."""
    names = {"openai"}
    for d in _format_search_dirs():
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".json"):
                    names.add(f[:-5])
    return sorted(names)


def detect_format(api_url: str) -> str:
    """Auto-detect format from URL. Returns 'openai' as default."""
    if "anthropic.com" in api_url:
        return "anthropic"
    return "openai"


# -----------------------------------------------------------------------
# Request builders
# -----------------------------------------------------------------------
def build_request_body(
    fmt: Dict[str, Any],
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int,
    stream: bool = False,
) -> dict:
    """Build the request JSON body according to the format's transform."""
    transform = fmt.get("request_transform", "openai")

    if transform == "anthropic":
        return _build_anthropic_body(messages, model, max_tokens, stream)
    else:
        return _build_openai_body(messages, model, max_tokens, stream)


def _build_openai_body(messages, model, max_tokens, stream) -> dict:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def _build_anthropic_body(messages, model, max_tokens, stream) -> dict:
    """
    Anthropic /v1/messages: system message is a top-level field,
    not part of the messages array.
    """
    system_text = ""
    chat_messages = []

    for msg in messages:
        if msg["role"] == "system":
            if system_text:
                system_text += "\n\n"
            system_text += msg["content"]
        else:
            chat_messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    body: dict = {
        "model": model,
        "messages": chat_messages,
        "max_tokens": max_tokens,
    }
    if system_text:
        body["system"] = system_text
    if stream:
        body["stream"] = True
    return body


# -----------------------------------------------------------------------
# Response extractors using dot-bracket paths
# -----------------------------------------------------------------------
def _extract_path(data: dict, path: str) -> Optional[str]:
    """
    Extract a value from nested dict/list using a dot-bracket path.

    "choices[0].message.content" → data["choices"][0]["message"]["content"]
    "content[0].text"            → data["content"][0]["text"]
    """
    obj: Any = data
    for part in _split_path(path):
        if obj is None:
            return None
        if isinstance(part, int):
            if isinstance(obj, list) and len(obj) > part:
                obj = obj[part]
            else:
                return None
        else:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None
    return str(obj) if obj is not None else None


def _split_path(path: str) -> list:
    """Split 'choices[0].message.content' into ['choices', 0, 'message', 'content']."""
    parts = []
    for segment in path.replace("]", "").split("."):
        if "[" in segment:
            key, idx = segment.split("[", 1)
            if key:
                parts.append(key)
            parts.append(int(idx))
        else:
            parts.append(segment)
    return parts


def extract_content(fmt: Dict[str, Any], response: dict) -> str:
    """Extract text content from a non-streaming response."""
    path = fmt.get("response_content", "choices[0].message.content")
    return _extract_path(response, path) or ""


def extract_stream_delta(fmt: Dict[str, Any], chunk: dict) -> str:
    """Extract text from a streaming delta chunk, respecting type filters."""
    type_field = fmt.get("stream_type_field")
    type_value = fmt.get("stream_type_value")
    if type_field and type_value:
        if chunk.get(type_field) != type_value:
            return ""

    path = fmt.get("stream_content", "choices[0].delta.content")
    return _extract_path(chunk, path) or ""


def is_stream_stop(fmt: Dict[str, Any], line: str, chunk: Optional[dict] = None) -> bool:
    """Check if a stream line/chunk signals completion."""
    stop = fmt.get("stream_stop", "[DONE]")

    if line.strip() == f"data: {stop}":
        return True

    if chunk and isinstance(chunk, dict):
        type_field = fmt.get("stream_type_field", "type")
        if chunk.get(type_field) == stop:
            return True

    return False


# -----------------------------------------------------------------------
# Header builder
# -----------------------------------------------------------------------
def build_headers(fmt: Dict[str, Any], api_key: str) -> dict:
    """Build HTTP headers for a request."""
    headers = {"Content-Type": "application/json"}

    if api_key:
        auth_style = fmt.get("auth", "bearer")
        if auth_style == "bearer":
            prefix = fmt.get("auth_prefix", "Bearer ")
            header_name = fmt.get("auth_header", "Authorization")
            headers[header_name] = f"{prefix}{api_key}"
        elif auth_style == "header":
            header_name = fmt.get("auth_header", "x-api-key")
            prefix = fmt.get("auth_prefix", "")
            headers[header_name] = f"{prefix}{api_key}"

    for k, v in fmt.get("auth_extra", {}).items():
        headers[k] = v

    return headers
