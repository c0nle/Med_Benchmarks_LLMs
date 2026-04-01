import requests
import json
import os
from urllib.parse import urlparse

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

    def ask_question(self, prompt):
        if self._openai_client is not None:
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "Du bist ein medizinischer Experte. Antworte präzise."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                }
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens

                response = self._openai_client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as e:
                return f"Error: {str(e)}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Du bist ein medizinischer Experte. Antworte präzise."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout_s,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
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
