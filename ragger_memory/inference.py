"""
Multi-endpoint inference client for LLM APIs.

Routes requests to the correct endpoint based on model name glob patterns.
API format differences (OpenAI vs Anthropic vs custom) handled by
schema-driven format definitions in api_formats.py.

Uses only urllib (no requests dependency).
"""

import fnmatch
import json
import logging
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any, Iterator

from . import api_formats

logger = logging.getLogger(__name__)


class Endpoint:
    """One inference endpoint with its URL, key, model patterns, and API format."""

    def __init__(self, name: str, api_url: str, api_key: str = "",
                 models: str = "*", format: str = "", max_context: int = 0,
                 max_tokens: int = 0):
        self.name = name
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.max_context = max_context  # 0 = unknown/unlimited
        self.max_tokens = max_tokens    # 0 = use client default
        self._patterns = [p.strip() for p in models.split(",") if p.strip()]
        # Format: explicit, auto-detected, or default
        self.format_name = format or api_formats.detect_format(self.api_url)
        self._fmt = api_formats.get_format(self.format_name)

    def matches(self, model: str) -> bool:
        """Check if this endpoint handles the given model name."""
        return any(fnmatch.fnmatch(model, p) for p in self._patterns)

    def headers(self) -> dict:
        """Build headers using the format's auth config."""
        return api_formats.build_headers(self._fmt, self.api_key)

    def request_url(self) -> str:
        """Build the full request URL using the format's path."""
        path = self._fmt.get("path", "/chat/completions")
        return f"{self.api_url}{path}"

    def build_body(self, messages, model, max_tokens, stream=False) -> dict:
        """Build request body using the format's transform."""
        return api_formats.build_request_body(
            self._fmt, messages, model, max_tokens, stream
        )

    def extract_content(self, response: dict) -> str:
        """Extract text from non-streaming response."""
        return api_formats.extract_content(self._fmt, response)

    def extract_delta(self, chunk: dict) -> str:
        """Extract text from streaming delta chunk."""
        return api_formats.extract_stream_delta(self._fmt, chunk)

    def is_stream_stop(self, line: str, chunk: dict = None) -> bool:
        """Check if stream is done."""
        return api_formats.is_stream_stop(self._fmt, line, chunk)

    def __repr__(self):
        return f"Endpoint({self.name!r}, {self.api_url!r}, format={self.format_name!r})"


class InferenceClient:
    """
    Multi-endpoint inference client.

    Routing: model name is matched against each endpoint's glob patterns.
    First match wins. Default endpoint (from [inference] section) matches
    everything as fallback.

    Config layout (INI):
        [inference]
        model = claude-sonnet-4-5
        max_tokens = 4096
        default = local

        [inference.local]
        api_url = http://localhost:1234/v1
        models = qwen/*, llama/*
        format = openai

        [inference.anthropic]
        api_url = https://api.anthropic.com/v1
        api_key = sk-ant-...
        models = claude-*
        format = anthropic
    """

    def __init__(
        self,
        endpoints: Optional[List[Endpoint]] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        max_tokens: int = 4096
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._endpoints: List[Endpoint] = []

        if endpoints:
            self._endpoints = endpoints
        elif api_url:
            self._endpoints = [Endpoint("default", api_url, api_key or "", "*")]

    def _resolve_endpoint(self, model: str) -> Endpoint:
        """Find the endpoint that handles this model."""
        for ep in self._endpoints:
            if ep.matches(model):
                return ep
        if self._endpoints:
            return self._endpoints[-1]
        raise RuntimeError("No inference endpoints configured")

    @classmethod
    def from_config(cls, cfg: dict) -> "InferenceClient":
        """Build from loaded config dict."""
        endpoints = []
        model = cfg.get("inference_model", "claude-sonnet-4-5")
        max_tokens = cfg.get("inference_max_tokens", 4096)
        default_name = cfg.get("inference_default", "")

        ep_list = cfg.get("inference_endpoints", [])
        for ep in ep_list:
            endpoints.append(Endpoint(
                name=ep.get("name", ""),
                api_url=ep.get("api_url", ""),
                api_key=ep.get("api_key", ""),
                models=ep.get("models", "*"),
                format=ep.get("format", ""),
                max_context=ep.get("max_context", 0),
                max_tokens=ep.get("max_tokens", 0),
            ))

        if not endpoints:
            api_url = cfg.get("inference_api_url", "")
            api_key = cfg.get("inference_api_key", "")
            if api_url:
                endpoints.append(Endpoint("default", api_url, api_key, "*"))

        if default_name and len(endpoints) > 1:
            named = [e for e in endpoints if e.name == default_name]
            others = [e for e in endpoints if e.name != default_name]
            endpoints = others + named

        return cls(endpoints=endpoints, model=model, max_tokens=max_tokens)

    def ensure_model_loaded(self, model: Optional[str] = None) -> Optional[str]:
        """
        Check if the model is loaded in a local inference engine. If not, trigger
        a load and return a status message. Returns None if already loaded or if
        the endpoint doesn't support model management.

        Currently supports LM Studio's /api/v1/models API.
        """
        use_model = model or self.model
        endpoint = self._resolve_endpoint(use_model)
        base_url = endpoint.api_url.rstrip("/")

        # Only attempt for local endpoints (LM Studio v1 API)
        # Derive the management API base from the OpenAI-compat URL
        # e.g. http://localhost:1234/v1 → http://localhost:1234/api/v1
        if "/v1" in base_url:
            mgmt_base = base_url.rsplit("/v1", 1)[0] + "/api/v1"
        else:
            return None  # not a recognized local engine

        try:
            # Check if model is loaded
            req = urllib.request.Request(f"{mgmt_base}/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())

            models = data.get("models", data.get("data", []))
            for m in models:
                key = m.get("key", m.get("id", ""))
                loaded = len(m.get("loaded_instances", [])) > 0
                if key == use_model and loaded:
                    return None  # already loaded

            # Model not loaded — trigger load
            logger.info(f"Model {use_model!r} not loaded, requesting load...")
            load_body = json.dumps({"model": use_model}).encode()
            load_req = urllib.request.Request(
                f"{mgmt_base}/models/load",
                data=load_body,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(load_req, timeout=120) as load_resp:
                load_data = json.loads(load_resp.read())
                load_time = load_data.get("load_time_seconds", "?")
                logger.info(f"Model {use_model!r} loaded in {load_time}s")
                return None  # loaded successfully

        except urllib.error.URLError as e:
            if "Connection refused" in str(e) or "urlopen error" in str(e):
                return f"Inference engine not reachable at {mgmt_base}"
            return f"Failed to load model {use_model}: {e}"
        except TimeoutError:
            return f"Model {use_model} load timed out"
        except Exception as e:
            logger.warning(f"ensure_model_loaded error: {e}")
            return None  # fail open — let chat attempt proceed

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """Send chat completion request, routed by model name."""
        use_model = model or self.model
        endpoint = self._resolve_endpoint(use_model)

        logger.debug(f"Routing {use_model!r} → {endpoint.name} ({endpoint.format_name})")

        url = endpoint.request_url()
        # Endpoint max_tokens overrides client default; explicit param overrides both
        effective_max = max_tokens or endpoint.max_tokens or self.max_tokens
        body = endpoint.build_body(
            messages, use_model, effective_max, stream
        )

        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers=endpoint.headers(), method="POST"
        )

        if stream:
            return self._stream_response(req, endpoint)
        else:
            return self._blocking_response(req, endpoint)

    def _blocking_response(self, req, endpoint: Endpoint) -> Dict[str, Any]:
        """Get non-streaming response."""
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise RuntimeError(f"Inference API error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Inference API connection failed: {e.reason}") from e

    def _stream_response(self, req, endpoint: Endpoint) -> Iterator[Dict[str, Any]]:
        """Stream SSE response, yield delta chunks."""
        try:
            with urllib.request.urlopen(req) as resp:
                buffer = ""
                for chunk in resp:
                    buffer += chunk.decode()
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            continue
                        if endpoint.is_stream_stop(line):
                            return
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                            if endpoint.is_stream_stop("", data):
                                return
                            yield data
                        except json.JSONDecodeError:
                            continue
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise RuntimeError(f"Inference API error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Inference API connection failed: {e.reason}") from e

    def extract_content(self, response: Dict[str, Any], model: Optional[str] = None) -> str:
        """Extract text content from non-streaming response."""
        use_model = model or self.model
        endpoint = self._resolve_endpoint(use_model)
        return endpoint.extract_content(response)

    def extract_delta(self, delta_chunk: Dict[str, Any], model: Optional[str] = None) -> str:
        """Extract text from streaming delta chunk."""
        use_model = model or self.model
        endpoint = self._resolve_endpoint(use_model)
        return endpoint.extract_delta(delta_chunk)
