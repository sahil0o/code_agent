"""The model the agent talks to. Any provider with an OpenAI-compatible API works:
Groq, OpenAI, OpenRouter, or a model on your own machine (Ollama, LM Studio, vLLM)."""
import os
import time

from openai import BadRequestError, OpenAI

# Known providers: where their API lives, which env variable holds the key, a sensible default model.
PROVIDERS = {
    "groq":       {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
                   "model": "openai/gpt-oss-120b"},
    "openai":     {"base_url": "https://api.openai.com/v1",      "key_env": "OPENAI_API_KEY",
                   "model": None},      # no default: model names change often, so choose one
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",   "key_env": "OPENROUTER_API_KEY",
                   "model": "openai/gpt-oss-120b"},
    "ollama":     {"base_url": "http://localhost:11434/v1",      "key_env": None,
                   "model": "qwen2.5-coder:7b"},
    "gemini":     {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": "GEMINI_API_KEY", "model": None},
}

# Error codes some providers return when the MODEL wrote a broken tool call or unreadable output.
# These are worth retrying; any other 400 error is a real bug and should crash.
MODEL_OUTPUT_ERRORS = ("tool_use_failed", "output_parse_failed")


class BadModelOutput(Exception):
    """The provider rejected what the model generated. The agent can ask the model to try again."""

    def __init__(self, code, failed_text):
        super().__init__(code)
        self.code = code
        self.failed_text = failed_text


class LLM:
    def __init__(self, provider="groq", model=None, base_url=None, api_key=None,
                 max_tokens=4000, timeout=60, max_retries=5):
        preset = PROVIDERS.get(provider)
        if preset is None and not base_url:
            raise ValueError(f"Unknown provider '{provider}'. Known: {', '.join(PROVIDERS)}. "
                             f"For any other OpenAI-compatible server, pass base_url.")
        preset = preset or {"base_url": None, "key_env": None, "model": None}

        self.provider = provider
        self.model = model or preset["model"]
        if not self.model:
            raise ValueError("No model given. Pass model=... (e.g. the name your server uses).")
        if api_key is None and preset["key_env"]:
            api_key = os.environ.get(preset["key_env"])
            if not api_key:
                raise ValueError(f"Set {preset['key_env']} to use {provider}.")
        self.max_tokens = max_tokens
        self.client = OpenAI(
            base_url=base_url or preset["base_url"],
            api_key=api_key or "not-needed",   # local servers ignore the key, but the SDK wants one
            timeout=timeout,
            max_retries=max_retries,           # rate limits and network blips are retried automatically
        )

    @classmethod
    def from_env(cls, provider=None, model=None, base_url=None):
        """Build from arguments, falling back to AGENT_PROVIDER / AGENT_MODEL / AGENT_BASE_URL."""
        return cls(
            provider=provider or os.environ.get("AGENT_PROVIDER", "groq"),
            model=model or os.environ.get("AGENT_MODEL") or None,       # None = the provider's default
            base_url=base_url or os.environ.get("AGENT_BASE_URL") or None,
        )

    def chat(self, messages, tools=None, **extra):
        """One model call. Returns (response, seconds). Raises BadModelOutput if the model's output was rejected."""
        started = time.time()
        if tools:
            extra["tools"] = tools   # only send "tools" when there are some; some servers reject tools=null
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, max_completion_tokens=self.max_tokens, **extra,
            )
        except BadRequestError as e:
            code = next((c for c in MODEL_OUTPUT_ERRORS if c in str(e)), None)
            if code is None:
                raise   # a real problem (bad model name, bad request): crash loudly
            body = e.body if isinstance(e.body, dict) else {}
            body = body.get("error", body)
            raise BadModelOutput(code, str(body.get("failed_generation", "(not provided)"))) from e
        return response, time.time() - started

    def __repr__(self):
        return f"LLM({self.provider}: {self.model})"