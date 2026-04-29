"""Ollama-backed optional text generation adapter."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from fullerene.models.base import ModelAdapter, ModelAdapterError


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"


class OllamaAdapter(ModelAdapter):
    def __init__(self, model_name: str, *, timeout: float = 10.0) -> None:
        cleaned_name = model_name.strip()
        if not cleaned_name:
            raise ValueError("Ollama model name must not be empty.")
        self.model_name = cleaned_name
        self.timeout = float(timeout)

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        request = urllib.request.Request(
            OLLAMA_GENERATE_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read().decode("utf-8")
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise ModelAdapterError(f"Ollama unavailable: {exc}") from exc

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ModelAdapterError("Ollama returned invalid JSON.") from exc

        generated = body.get("response")
        if not isinstance(generated, str):
            raise ModelAdapterError("Ollama response did not include generated text.")
        return generated
