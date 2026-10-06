import requests
import json
import os
import re
import threading
import time
from urllib.parse import urlparse


class ServerUnavailableError(RuntimeError):
    """Raised after too many consecutive transient failures, so a benchmark stops early."""

class MedicalLLMClient:
    def __init__(self, config):
        self.client_type = (config.get("server", {}).get("client") or "requests").strip()
        self.base_url = self._normalize_base_url(config["server"]["url"])
        self.url = self._normalize_chat_completions_url(config["server"]["url"])
        self.model = config['server']['model_name']
        self.verify_ssl = config.get("server", {}).get("verify_ssl", True)
        self.timeout_s = config.get("server", {}).get("timeout_s", 60)
        self.temperature = config.get("benchmark_settings", {}).get("temperature", 0)
        self.max_tokens = config.get("benchmark_settings", {}).get("max_tokens", None)
        # Extra request fields, e.g. {"chat_template_kwargs": {"enable_thinking": false}}
        # for reasoning models whose thinking would otherwise consume max_tokens.
        self.extra_body = config.get("server", {}).get("extra_body") or {}
        # Retries for transient failures (timeouts, connection errors, 429, 5xx)
        self.max_retries = int(config.get("server", {}).get("max_retries", 3))
        self.retry_backoff_s = float(config.get("server", {}).get("retry_backoff_s", 10))
        self.max_consecutive_errors = int(config.get("server", {}).get("max_consecutive_errors", 20))
        self._consecutive_errors = 0
        self._error_lock = threading.Lock()
        self.headers = {"Content-Type": "application/json"}
        api_key = (config.get("server", {}).get("api_key") or "").strip()
        if not api_key:
            env_name = (config.get("server", {}).get("api_key_env") or "MED_SERVER_API_KEY").strip()
            api_key = (os.getenv(env_name) or "").strip()
        self.api_key = api_key
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self._openai_client = None
        if self.client_type.lower() in {"openai", "openai_sdk", "openai-sdk"}:
            try:
                import httpx
                from openai import OpenAI
            except Exception as e:
                raise RuntimeError(
                    "OpenAI SDK ist nicht installiert. Installiere es mit `pip install openai` "
                    "oder setze in config.yaml `server.client: requests`."
                ) from e

            # OpenAI SDK benötigt base_url bis inkl. /v1
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

    def _build_messages(self, user_content):
        """Build the standard message list with system prompt."""
        return [
            {"role": "system", "content": "You are a medical expert in diagnostic imaging. Answer concisely and in English."},
            {"role": "user", "content": user_content},
        ]

    def _call_openai(self, messages):
        """Call via OpenAI SDK and return content string or Error: string."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body
            response = self._openai_client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            return self._content_or_error(choice.message.content, choice.finish_reason)
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def _content_or_error(content, finish_reason) -> str:
        if content is None or not str(content).strip():
            return f"Error: empty response (finish_reason={finish_reason})"
        return content

    def _call_requests(self, messages):
        """Call via raw requests and return content string or Error: string."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
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
            choice = response.json()["choices"][0]
            return self._content_or_error(choice["message"].get("content"), choice.get("finish_reason"))
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else "?"
            detail = ""
            if resp is not None:
                try:
                    detail = json.dumps(resp.json(), ensure_ascii=False)
                except Exception:
                    detail = (resp.text or "").strip()
            if len(detail) > 1000:
                detail = detail[:1000] + "…"
            return f"Error: HTTP {status} {detail}".strip()
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def _is_transient(answer: str) -> bool:
        """Errors worth retrying: not empty answers and not client errors (400/401/403/404/422)."""
        if not answer.startswith("Error:") or answer.startswith("Error: empty response"):
            return False
        return not re.search(r"(Error code:|HTTP)\s*(400|401|403|404|422)\b", answer)

    def _send(self, messages) -> str:
        call = self._call_openai if self._openai_client is not None else self._call_requests
        for attempt in range(self.max_retries + 1):
            answer = call(messages)
            if not self._is_transient(answer) or attempt == self.max_retries:
                break
            time.sleep(self.retry_backoff_s * 2 ** attempt)

        with self._error_lock:
            if self._is_transient(answer):
                self._consecutive_errors += 1
                if self._consecutive_errors >= self.max_consecutive_errors:
                    raise ServerUnavailableError(
                        f"{self._consecutive_errors} consecutive request failures, last: {answer[:300]}"
                    )
            elif not answer.startswith("Error:"):
                self._consecutive_errors = 0
        return answer

    def ask_question(self, prompt: str, system_prompt: str = None) -> str:
        """Send a text-only prompt and return the model's answer."""
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = self._build_messages(prompt)
        return self._send(messages)

    def ask_with_image(self, prompt: str, image_b64: str, image_format: str = "jpeg") -> str:
        """
        Send a prompt together with a base64-encoded image (VLM / multimodal).

        Parameters
        ----------
        prompt       : The text question / instruction.
        image_b64    : Base64-encoded image bytes (JPEG or PNG).
        image_format : "jpeg" or "png"  (default: "jpeg").

        Returns the model's answer or "Error: ..." on failure.
        Requires the target model to be a vision-capable LLM.
        """
        return self.ask_with_images(prompt, [image_b64], image_format)

    def ask_with_images(self, prompt: str, images_b64: list, image_format: str = "jpeg") -> str:
        """Like ask_with_image, but sends all images (in order) before the text prompt."""
        media_type = f"image/{'jpeg' if image_format.lower() in ('jpg', 'jpeg') else image_format.lower()}"
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}
            for b64 in images_b64
        ]
        user_content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": "You are a medical expert in diagnostic imaging. Answer concisely and in English."},
            {"role": "user", "content": user_content},
        ]
        return self._send(messages)
