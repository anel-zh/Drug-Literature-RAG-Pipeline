from __future__ import annotations

from typing import Dict, Generator, Optional
import ollama


class LocalLLM:
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def generate(
        self,
        system: str,
        user: str,
        options: Optional[Dict] = None,
    ) -> str:
        """Non-streaming (kept for fallback)."""
        resp = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options=options or {
                "temperature": 0.1,
                "num_predict": 200,
                "top_p": 0.9,
            },
        )
        return resp["message"]["content"]

    def stream_lines(
        self,
        system: str,
        user: str,
        options: Optional[Dict] = None,
    ) -> Generator[str, None, str]:
        """
        Streams response and yields completed lines as soon as they form.
        Returns full text at the end (via StopIteration.value).
        """
        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options=options or {
                "temperature": 0.1,
                "num_predict": 200,
                "top_p": 0.9,
            },
            stream=True,
        )

        buffer = ""
        full = ""

        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if not token:
                continue

            full += token
            buffer += token

            # emit complete lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                yield line + "\n"

        # emit remaining partial line (if any)
        if buffer.strip():
            yield buffer

        return full