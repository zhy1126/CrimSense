"""Model calls and validation of the requested JSON output."""

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from simulation.io import dumps


def strict_json(value):
    def pairs(items):
        out = {}
        for key, item in items:
            if key in out:
                raise ValueError("Duplicate JSON key: " + key)
            out[key] = item
        return out

    def invalid(value):
        raise ValueError("Nonfinite JSON value")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)


class ContextLimit:
    def __init__(self, directory, limit=8192):
        from tokenizers import Tokenizer
        from jinja2 import Environment, StrictUndefined

        root = Path(directory)
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.config = json.loads((root / "tokenizer_config.json").read_text())
        self.limit = limit
        env = Environment(undefined=StrictUndefined)

        def raise_exception(message):
            raise ValueError(message)

        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string(self.config["chat_template"])

    def __call__(self, body):
        rendered = self.template.render(
            messages=body["messages"],
            tools=None,
            documents=None,
            add_generation_prompt=True,
            bos_token=self.config.get("bos_token") or "",
            eos_token=self.config.get("eos_token") or "",
        )
        count = len(self.tokenizer.encode(rendered, add_special_tokens=False).ids)
        if count + body["max_tokens"] > self.limit:
            raise ValueError(
                f'Context overflow: {count} input + {body["max_tokens"]} output tokens'
            )
        return count


class Client:
    """One request per decision stage, with explicit JSON output validation."""

    def __init__(
        self, *, transport=None, context_check=None, base_url=None, api_key=None
    ):
        self.check = context_check
        self.mode = "offline_test" if transport is not None else "live"
        self.endpoint = (base_url or os.environ.get("CRIMSENSE_BASE_URL", "")).rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("CRIMSENSE_API_KEY")
        if transport is None:
            if context_check is None:
                raise ValueError("The matching model tokenizer is required")
            if not self.endpoint.startswith(("https://", "http://")):
                raise ValueError(
                    "Set CRIMSENSE_BASE_URL to an OpenAI-compatible /v1 endpoint"
                )
        self.transport = transport or self._http

    def _http(self, body):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(
            self.endpoint + "/chat/completions",
            data=dumps(body).encode(),
            headers=headers,
        )
        with urlopen(request, timeout=120) as response:
            return strict_json(response.read().decode())

    def call(self, key, job):
        if self.check:
            self.check(job["body"])
        envelope = self.transport(job["body"])
        choices = envelope.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError("Expected one complete model answer: " + key)
        return job["validate"](strict_json(choices[0]["message"]["content"]))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
