from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    raw: dict[str, Any]
    runtime_seconds: float


@dataclass(frozen=True)
class ImagePayload:
    data: str
    mime_type: str = "image/png"


def create_native_client(
    *,
    provider: str,
    base_url: str,
    api_key: str | None = None,
    api_key_env: str | None = None,
    timeout_seconds: int = 120,
) -> NativeClient:
    normalized = provider.strip().lower().replace("_", "-").replace(" ", "-")
    if normalized == "ollama":
        return OllamaNativeClient(base_url=base_url, timeout_seconds=timeout_seconds)
    if normalized in {"lmstudio", "lm-studio"}:
        return LMStudioNativeClient(
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
        )
    if normalized == "omlx":
        return OmlxNativeClient(
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError("provider must be one of: ollama, lmstudio, omlx")


class NativeClient:
    provider_name = "native"

    def planned_endpoint(self) -> str:
        raise NotImplementedError

    def models_endpoint(self) -> str:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        raise NotImplementedError

    def require_model(self, model: str) -> None:
        models = self.list_models()
        if model in models:
            return
        available = ", ".join(models[:20]) if models else "none"
        if len(models) > 20:
            available += ", ..."
        raise RuntimeError(
            f"Model `{model}` was not returned by {self.models_endpoint()}. "
            f"Available model id(s): {available}. Use one of those ids or start/load "
            "the model in the local server."
        )

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
        request_extra: dict[str, Any] | None = None,
        images: list[ImagePayload | dict[str, str]] | None = None,
    ) -> GenerationResponse:
        raise NotImplementedError

    def reset_model_runtime(
        self,
        model: str,
        request_extra: dict[str, Any] | None = None,
    ) -> str | None:
        return None

    def _request_json(
        self,
        url: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], float]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{url} returned HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(self._connection_error(url, exc)) from exc
        except (http.client.IncompleteRead, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"{self.provider_name} native server dropped the response from {url}: {exc}. "
                "The model may have crashed, timed out, or exceeded the server's generation limits."
            ) from exc
        runtime = time.perf_counter() - start
        return json.loads(raw_body), runtime

    def _connection_error(self, url: str, exc: urllib.error.URLError) -> str:
        reason = getattr(exc, "reason", exc)
        return (
            f"Could not connect to {self.provider_name} native server at {url}: {reason}. "
            f"Check that the local server is running and that --base-url is correct "
            f"(current base URL: {self.base_url})."
        )


class OllamaNativeClient(NativeClient):
    provider_name = "Ollama"

    def __init__(self, *, base_url: str, timeout_seconds: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def planned_endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    def models_endpoint(self) -> str:
        return f"{self.base_url}/api/tags"

    def list_models(self) -> list[str]:
        payload, _ = self._request_json(self.models_endpoint())
        models: list[str] = []
        for item in payload.get("models", []):
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if name:
                models.append(str(name))
        return models

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
        request_extra: dict[str, Any] | None = None,
        images: list[ImagePayload | dict[str, str]] | None = None,
    ) -> GenerationResponse:
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if top_p is not None:
            options["top_p"] = top_p
        if stop is not None:
            options["stop"] = stop
        if seed is not None:
            options["seed"] = seed

        message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            message["images"] = [_image_data(image) for image in images]

        payload: dict[str, Any] = {
            "model": model,
            "messages": [message],
            "stream": False,
            "options": options,
        }
        think = _ollama_think(reasoning_effort)
        if think is not None:
            payload["think"] = think
        if response_format is not None:
            payload["format"] = response_format
        if request_extra:
            extra = dict(request_extra)
            extra_options = extra.pop("options", None)
            payload.update(extra)
            if isinstance(extra_options, dict):
                payload.setdefault("options", {}).update(extra_options)

        raw, runtime = self._request_json(self.planned_endpoint(), method="POST", payload=payload)
        message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
        content = str(message.get("content") or "")
        usage = _ollama_usage(raw)
        if usage:
            raw.setdefault("usage", usage)
        return GenerationResponse(text=content, raw=raw, runtime_seconds=runtime)


class LMStudioNativeClient(NativeClient):
    provider_name = "LM Studio"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout_seconds: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self.timeout_seconds = timeout_seconds
        self._models_payload: list[dict[str, Any]] | None = None

    def planned_endpoint(self) -> str:
        return f"{self.base_url}/api/v1/chat"

    def chat_completions_endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def models_endpoint(self) -> str:
        return f"{self.base_url}/api/v1/models"

    def list_models(self) -> list[str]:
        models = self._model_payloads()
        ids: list[str] = []
        for item in models:
            for key in _lmstudio_model_ids(item):
                if key not in ids:
                    ids.append(key)
        return ids

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
        request_extra: dict[str, Any] | None = None,
        images: list[ImagePayload | dict[str, str]] | None = None,
    ) -> GenerationResponse:
        if _lmstudio_uses_no_think_system_prompt(model, reasoning_effort) and not images:
            return self._generate_chat_completion_no_think(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                seed=seed,
                response_format=response_format,
            )

        input_payload: str | list[dict[str, Any]]
        if images:
            input_payload = [{"type": "text", "content": prompt}]
            input_payload.extend(
                {"type": "image", "data_url": _image_data_url(image)} for image in images
            )
        else:
            input_payload = prompt

        payload: dict[str, Any] = {
            "model": model,
            "input": input_payload,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "stream": False,
            "store": False,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        reasoning = self._reasoning_setting(model, reasoning_effort)
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if request_extra:
            payload.update(request_extra)

        raw, runtime = self._request_json(
            self.planned_endpoint(),
            method="POST",
            payload=payload,
            headers=self._headers(),
        )
        text = _lmstudio_text(raw)
        usage = _lmstudio_usage(raw)
        if usage:
            raw.setdefault("usage", usage)
        return GenerationResponse(text=text, raw=raw, runtime_seconds=runtime)

    def _generate_chat_completion_no_think(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float | None,
        stop: list[str] | None,
        seed: int | None,
        response_format: dict[str, Any] | None,
    ) -> GenerationResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
        if response_format is not None:
            payload["response_format"] = response_format

        raw, runtime = self._request_json(
            self.chat_completions_endpoint(),
            method="POST",
            payload=payload,
            headers=self._headers(),
        )
        message = _first_chat_completion_message(raw)
        text = str(message.get("content") or "")
        return GenerationResponse(text=text, raw=raw, runtime_seconds=runtime)

    def reset_model_runtime(
        self,
        model: str,
        request_extra: dict[str, Any] | None = None,
    ) -> str | None:
        self._models_payload = None
        instance_id = None
        for item in self._model_payloads():
            if model not in _lmstudio_model_ids(item):
                continue
            for instance in item.get("loaded_instances") or []:
                if isinstance(instance, dict) and instance.get("id"):
                    instance_id = str(instance["id"])
                    break
            if instance_id:
                break
        if not instance_id:
            return None
        self._request_json(
            f"{self.base_url}/api/v1/models/unload",
            method="POST",
            payload={"instance_id": instance_id},
            headers=self._headers(),
        )
        load_payload: dict[str, Any] = {"model": model}
        if request_extra:
            load_payload.update(request_extra)
        self._request_json(
            f"{self.base_url}/api/v1/models/load",
            method="POST",
            payload=load_payload,
            headers=self._headers(),
        )
        self._models_payload = None
        return instance_id

    def _model_payloads(self) -> list[dict[str, Any]]:
        if self._models_payload is None:
            payload, _ = self._request_json(self.models_endpoint(), headers=self._headers())
            models = payload.get("models", [])
            self._models_payload = [item for item in models if isinstance(item, dict)]
        return self._models_payload

    def _reasoning_setting(self, model: str, value: str | None) -> str | None:
        normalized = _lmstudio_reasoning(value)
        if normalized is None:
            return None
        info = self._model_info(model)
        reasoning = ((info.get("capabilities") or {}).get("reasoning") or {})
        allowed = reasoning.get("allowed_options") or []
        if normalized in allowed:
            return normalized
        if normalized == "off":
            return None
        raise RuntimeError(
            f"Model `{model}` does not expose native LM Studio reasoning={normalized!r}. "
            f"Allowed reasoning option(s): {', '.join(allowed) if allowed else 'none'}."
        )

    def _model_info(self, model: str) -> dict[str, Any]:
        for item in self._model_payloads():
            if model in _lmstudio_model_ids(item):
                return item
        return {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def _ollama_think(value: str | None) -> bool | str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "off", "false", "0"}:
        return False
    if normalized in {"true", "on", "1"}:
        return True
    if normalized in {"low", "medium", "high"}:
        return normalized
    return normalized


def _ollama_usage(raw: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = raw.get("prompt_eval_count")
    completion_tokens = raw.get("eval_count")
    usage: dict[str, Any] = {}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if prompt_tokens is not None and completion_tokens is not None:
        usage["total_tokens"] = prompt_tokens + completion_tokens
    return usage


def _lmstudio_model_ids(item: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("key", "selected_variant"):
        value = item.get(key)
        if value:
            ids.append(str(value))
    for instance in item.get("loaded_instances") or []:
        if isinstance(instance, dict) and instance.get("id"):
            ids.append(str(instance["id"]))
    for variant in item.get("variants") or []:
        if variant:
            ids.append(str(variant))
    return ids


def _lmstudio_reasoning(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "off", "false", "0"}:
        return "off"
    if normalized in {"true", "on", "1"}:
        return "on"
    if normalized in {"low", "medium", "high"}:
        return normalized
    return normalized


def _lmstudio_uses_no_think_system_prompt(model: str, reasoning_effort: str | None) -> bool:
    model_id = model.lower()
    uses_prompt_switch = "smollm3" in model_id or "nemotron-3-nano" in model_id
    if not uses_prompt_switch:
        return False
    return _lmstudio_reasoning(reasoning_effort) == "off"


def _lmstudio_text(raw: dict[str, Any]) -> str:
    output = raw.get("output") or []
    parts = [
        str(item.get("content") or "")
        for item in output
        if isinstance(item, dict) and item.get("type") == "message"
    ]
    return "".join(parts)


def _first_chat_completion_message(raw: dict[str, Any]) -> dict[str, Any]:
    choices = raw.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message") or {}
    return message if isinstance(message, dict) else {}


def _lmstudio_usage(raw: dict[str, Any]) -> dict[str, Any]:
    stats = raw.get("stats")
    if not isinstance(stats, dict):
        return {}
    prompt_tokens = stats.get("input_tokens")
    completion_tokens = stats.get("total_output_tokens")
    usage: dict[str, Any] = {}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if prompt_tokens is not None and completion_tokens is not None:
        usage["total_tokens"] = prompt_tokens + completion_tokens
    reasoning_tokens = stats.get("reasoning_output_tokens")
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return usage


class OmlxNativeClient(NativeClient):
    provider_name = "oMLX"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout_seconds: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self.timeout_seconds = timeout_seconds

    @property
    def _v1_base_url(self) -> str:
        return self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"

    def planned_endpoint(self) -> str:
        return f"{self._v1_base_url}/chat/completions"

    def models_endpoint(self) -> str:
        return f"{self._v1_base_url}/models"

    def list_models(self) -> list[str]:
        payload, _ = self._request_json(self.models_endpoint(), headers=self._headers())
        models: list[str] = []
        for item in payload.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
        return models

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
        request_extra: dict[str, Any] | None = None,
        images: list[ImagePayload | dict[str, str]] | None = None,
    ) -> GenerationResponse:
        content: str | list[dict[str, Any]]
        if images:
            content = [{"type": "text", "text": prompt}]
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image)},
                }
                for image in images
            )
        else:
            content = prompt

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
        if response_format is not None:
            payload["response_format"] = response_format

        chat_template_kwargs = _omlx_chat_template_kwargs(reasoning_effort)
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs

        if request_extra:
            extra = dict(request_extra)
            extra_ct_kwargs = extra.pop("chat_template_kwargs", None)
            payload.update(extra)
            if isinstance(extra_ct_kwargs, dict):
                payload.setdefault("chat_template_kwargs", {}).update(extra_ct_kwargs)

        raw, runtime = self._request_json(
            self.planned_endpoint(),
            method="POST",
            payload=payload,
            headers=self._headers(),
        )
        text = _openai_chat_completion_text(raw)
        return GenerationResponse(text=text, raw=raw, runtime_seconds=runtime)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def _omlx_chat_template_kwargs(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    normalized = value.strip().lower()
    if normalized in {"", "none", "off", "false", "0"}:
        return {"enable_thinking": False}
    if normalized in {"true", "on", "1"}:
        return {"enable_thinking": True}
    if normalized in {"low", "medium", "high"}:
        return {"enable_thinking": True, "reasoning_effort": normalized}
    return {"enable_thinking": True, "reasoning_effort": normalized}


def _openai_chat_completion_text(raw: dict[str, Any]) -> str:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "")


def _image_data(image: ImagePayload | dict[str, str]) -> str:
    if isinstance(image, ImagePayload):
        return image.data
    return str(image.get("data") or "")


def _image_mime_type(image: ImagePayload | dict[str, str]) -> str:
    if isinstance(image, ImagePayload):
        return image.mime_type
    return str(image.get("mime_type") or "image/png")


def _image_data_url(image: ImagePayload | dict[str, str]) -> str:
    data = _image_data(image)
    if data.startswith("data:"):
        return data
    return f"data:{_image_mime_type(image)};base64,{data}"
