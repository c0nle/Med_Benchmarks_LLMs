import requests
import json
import os
import re
import threading
import time
from urllib.parse import urlparse


DEFAULT_SYSTEM_PROMPT = "You are a medical expert in diagnostic imaging. Answer concisely and in English."

# HTTP status codes that will not go away by retrying and make every further
# request fail as well (wrong key, no permission, wrong URL / unknown model).
_FATAL_STATUS = (401, 403, 404)
# Client errors that concern only the single request (e.g. prompt too long).
_CLIENT_STATUS = (400, 401, 403, 404, 422)


class ServerUnavailableError(RuntimeError):
    """Raised after too many consecutive transient failures, so a benchmark stops early."""


class ConfigurationError(ServerUnavailableError):
    """Server rejects the request itself (401/403/404, unknown model): stop at once.

    Subclass of ServerUnavailableError so every caller that stops on a server
    outage also stops on a configuration error.
    """


def _status_of(answer: str):
    """HTTP status code embedded in an "Error: ..." string (requests or OpenAI SDK format)."""
    m = re.match(r"Error:\s*(?:Error code:|HTTP)\s*(\d{3})\b", answer or "")
    return int(m.group(1)) if m else None


class MedicalLLMClient:
    def __init__(self, config):
        server = config.get("server", {})
        settings = config.get("benchmark_settings", {}) or {}
        self.client_type = (server.get("client") or "requests").strip()
        self.base_url = self._normalize_base_url(server["url"])
        self.url = self._normalize_chat_completions_url(server["url"])
        self.model = server['model_name']
        self.verify_ssl = server.get("verify_ssl", True)
        self.timeout_s = server.get("timeout_s", 60)
        self.temperature = settings.get("temperature", 0)
        self.max_tokens = settings.get("max_tokens", None)
        # Sampling seed sent with every request (null in the config disables it).
        seed = server.get("seed", 42)
        self.seed = int(seed) if seed is not None else None
        # Extra request fields, e.g. {"chat_template_kwargs": {"enable_thinking": false}}
        # for reasoning models whose thinking would otherwise consume max_tokens.
        self.extra_body = server.get("extra_body") or {}
        # Retries for transient failures (timeouts, connection errors, 429, 5xx).
        # Worst case before a benchmark stops:
        #   max_consecutive_errors * ((max_retries + 1) * timeout_s + backoff)
        self.max_retries = int(server.get("max_retries", 2))
        self.retry_backoff_s = float(server.get("retry_backoff_s", 5))
        self.max_consecutive_errors = int(server.get("max_consecutive_errors", 10))
        self._consecutive_errors = 0
        self._error_lock = threading.Lock()
        self._local = threading.local()
        # Model ids reported by the server in completion responses (for run_info)
        self.reported_models = set()
        self.headers = {"Content-Type": "application/json"}
        api_key = (server.get("api_key") or "").strip()
        if not api_key:
            env_name = (server.get("api_key_env") or "MED_SERVER_API_KEY").strip()
            api_key = (os.getenv(env_name) or "").strip()
        self.api_key = api_key
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self._openai_client = None
        if self.client_type.lower() in {"openai", "openai_sdk", "openai-sdk"}:
            try:
                import httpx
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "The OpenAI SDK is not installed. Install it with `pip install openai` "
                    "or set `server.client: requests` in the config."
                ) from e

            # The OpenAI SDK expects base_url up to and including /v1
            self._openai_client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "EMPTY",
                http_client=httpx.Client(verify=self.verify_ssl, timeout=self.timeout_s),
                max_retries=0,  # retries are handled in _send
            )

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        """
        Normalizes to an OpenAI-compatible base URL ending in /v1.
        Accepts base URLs and full /chat/completions endpoint URLs.
        """
        url = (url or "").strip()
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid server.url: {url!r}")

        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            return url[: -len("/chat/completions")].rstrip("/")
        if path in ("", "/"):
            return url.rstrip("/") + "/v1"
        return url.rstrip("/")

    @staticmethod
    def _normalize_chat_completions_url(url: str) -> str:
        """
        Accepts either a base URL (e.g. http://host:port) or a full endpoint URL.
        If no path is provided, defaults to /v1/chat/completions (OpenAI-compatible servers).
        """
        url = (url or "").strip()
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid server.url: {url!r}")

        if parsed.path in ("", "/"):
            return url.rstrip("/") + "/v1/chat/completions"
        if parsed.path.rstrip("/") == "/v1":
            return url.rstrip("/") + "/chat/completions"
        return url

    # -- per-thread metadata of the last call -----------------------------------

    @property
    def last_meta(self) -> dict:
        """Metadata of the most recent ask_* call made in the current thread.

        Keys: finish_reason (str|None, "error" if the call returned "Error: ..."),
        prompt_tokens, completion_tokens (int|None), model (str|None, as returned
        by the server), latency_s (float, last attempt), attempts (int),
        server_finish_reason (raw finish_reason of the server, also on errors).
        """
        meta = getattr(self._local, "meta", None)
        return dict(meta) if meta else self._empty_meta()

    @staticmethod
    def _empty_meta() -> dict:
        return {"finish_reason": None, "prompt_tokens": None, "completion_tokens": None,
                "model": None, "latency_s": 0.0, "attempts": 0, "server_finish_reason": None}

    def _record_response(self, finish_reason, usage, model):
        """Called by the transport methods when the server sent a parseable response."""
        meta = getattr(self._local, "meta", None)
        if meta is None:
            meta = self._local.meta = self._empty_meta()

        def _usage(key):
            value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
            return int(value) if isinstance(value, (int, float)) else None

        meta["server_finish_reason"] = finish_reason
        meta["prompt_tokens"] = _usage("prompt_tokens")
        meta["completion_tokens"] = _usage("completion_tokens")
        meta["model"] = model
        if model:
            with self._error_lock:
                self.reported_models.add(str(model))

    # -- transport ----------------------------------------------------------------

    def _build_messages(self, user_content, system_prompt: str = None):
        """Build the standard message list with system prompt."""
        return [
            {"role": "system", "content": system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _request_fields(self) -> dict:
        fields = {"model": self.model, "temperature": self.temperature}
        if self.max_tokens is not None:
            fields["max_tokens"] = self.max_tokens
        if self.seed is not None:
            fields["seed"] = self.seed
        return fields

    def _call_openai(self, messages):
        """Call via OpenAI SDK and return content string or Error: string."""
        import openai
        import httpx
        kwargs = {**self._request_fields(), "messages": messages}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        try:
            response = self._openai_client.chat.completions.create(**kwargs)
        except openai.APIStatusError as e:
            # same format as the requests path, so status-based handling works for both
            detail = str(e)
            if len(detail) > 1000:
                detail = detail[:1000] + "…"
            return f"Error: HTTP {e.status_code} {detail}".strip()
        except (openai.APIError, httpx.HTTPError) as e:
            # APIConnectionError / APITimeoutError
            return f"Error: {str(e)}"
        try:
            choice = response.choices[0]
            content, finish_reason = choice.message.content, choice.finish_reason
        except (AttributeError, IndexError, TypeError) as e:
            return f"Error: malformed response ({type(e).__name__}: {e})"
        self._record_response(finish_reason, getattr(response, "usage", None), getattr(response, "model", None))
        return self._content_or_error(content, finish_reason)

    @staticmethod
    def _content_or_error(content, finish_reason) -> str:
        if content is None or not str(content).strip():
            return f"Error: empty response (finish_reason={finish_reason})"
        return content

    def _call_requests(self, messages):
        """Call via raw requests and return content string or Error: string."""
        payload = {**self._request_fields(), "messages": messages}
        payload.update(self.extra_body)
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout_s,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else "?"
            detail = ""
            if resp is not None:
                try:
                    detail = json.dumps(resp.json(), ensure_ascii=False)
                except ValueError:
                    detail = (resp.text or "").strip()
            if len(detail) > 1000:
                detail = detail[:1000] + "…"
            return f"Error: HTTP {status} {detail}".strip()
        except requests.RequestException as e:
            return f"Error: {str(e)}"
        try:
            data = response.json()
            choice = data["choices"][0]
            content, finish_reason = choice["message"].get("content"), choice.get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            return f"Error: malformed response ({type(e).__name__}: {e})"
        self._record_response(finish_reason, data.get("usage") or {}, data.get("model"))
        return self._content_or_error(content, finish_reason)

    @staticmethod
    def _is_transient(answer: str) -> bool:
        """Errors worth retrying: not empty answers and not client errors (400/401/403/404/422)."""
        if not answer.startswith("Error:") or answer.startswith("Error: empty response"):
            return False
        return _status_of(answer) not in _CLIENT_STATUS

    def _send(self, messages) -> str:
        call = self._call_openai if self._openai_client is not None else self._call_requests
        meta = self._empty_meta()
        self._local.meta = meta
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            answer = call(messages)
            meta["latency_s"] = round(time.monotonic() - t0, 3)
            meta["attempts"] = attempt + 1
            if not self._is_transient(answer) or attempt == self.max_retries:
                break
            time.sleep(self.retry_backoff_s * 2 ** attempt)

        is_error = answer.startswith("Error:")
        meta["finish_reason"] = "error" if is_error else meta["server_finish_reason"]

        if is_error and _status_of(answer) in _FATAL_STATUS:
            raise ConfigurationError(
                f"server rejected the request ({answer[:300]}). Check server.url, the API key "
                f"and server.model_name={self.model!r}."
            )

        with self._error_lock:
            if self._is_transient(answer):
                self._consecutive_errors += 1
                if self._consecutive_errors >= self.max_consecutive_errors:
                    raise ServerUnavailableError(
                        f"{self._consecutive_errors} consecutive request failures, last: {answer[:300]}"
                    )
            elif not is_error:
                self._consecutive_errors = 0
        return answer

    # -- health check -------------------------------------------------------------

    def list_models(self):
        """GET <base_url>/models. Returns the list of model ids, or None if the
        endpoint is unreachable or returns no usable list (warn and continue).
        Raises ConfigurationError on 401/403 (key rejected)."""
        url = self.base_url.rstrip("/") + "/models"
        try:
            resp = requests.get(url, headers=self.headers, timeout=min(float(self.timeout_s), 30.0),
                                verify=self.verify_ssl)
        except requests.RequestException as e:
            print(f"  WARNING: health check {url} failed: {e}")
            return None
        if resp.status_code in (401, 403):
            raise ConfigurationError(f"health check {url}: HTTP {resp.status_code} – API key rejected")
        if resp.status_code != 200:
            print(f"  WARNING: health check {url}: HTTP {resp.status_code}")
            return None
        try:
            return [str(m["id"]) for m in resp.json()["data"]]
        except (ValueError, KeyError, TypeError) as e:
            print(f"  WARNING: health check {url}: unexpected response ({e})")
            return None

    def health_check(self, label: str = "model"):
        """Check that the configured model is served. Returns the model list (or None)."""
        models = self.list_models()
        if models is not None and self.model not in models:
            raise ConfigurationError(
                f"{label} {self.model!r} is not served by {self.base_url}. "
                f"Available: {', '.join(sorted(models)) or '(none)'}"
            )
        return models

    # -- public API ---------------------------------------------------------------

    def ask_question(self, prompt: str, system_prompt: str = None) -> str:
        """Send a text-only prompt and return the model's answer.

        system_prompt=None uses the default system prompt (DEFAULT_SYSTEM_PROMPT)."""
        return self._send(self._build_messages(prompt, system_prompt))

    def ask_with_image(self, prompt: str, image_b64: str, image_format: str = "jpeg",
                       system_prompt: str = None) -> str:
        """
        Send a prompt together with a base64-encoded image (VLM / multimodal).

        Parameters
        ----------
        prompt        : The text question / instruction.
        image_b64     : Base64-encoded image bytes (JPEG or PNG).
        image_format  : "jpeg" or "png"  (default: "jpeg").
        system_prompt : None = default system prompt.

        Returns the model's answer or "Error: ..." on failure.
        Requires the target model to be a vision-capable LLM.
        """
        return self.ask_with_images(prompt, [image_b64], image_format, system_prompt=system_prompt)

    def ask_with_images(self, prompt: str, images_b64: list, image_format: str = "jpeg",
                        system_prompt: str = None) -> str:
        """Like ask_with_image, but sends all images (in order) before the text prompt."""
        media_type = f"image/{'jpeg' if image_format.lower() in ('jpg', 'jpeg') else image_format.lower()}"
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}
            for b64 in images_b64
        ]
        user_content.append({"type": "text", "text": prompt})
        return self._send(self._build_messages(user_content, system_prompt))
