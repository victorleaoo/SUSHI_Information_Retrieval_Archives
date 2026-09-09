import hashlib
import json
import os
import time
from datetime import datetime

import openai
from dotenv import load_dotenv

load_dotenv()


class LLMRunner:
    """
    Runs LLM calls through an OpenAI-compatible endpoint (the UnB IA-Router
    LiteLLM proxy by default), with disk caching and per-call logging.

    Each unique prompt (SHA1 of `system_prompt + prompt`) is cached as a
    single JSON file under `cache_dir` containing the full call record
    (prompt, token counts, raw response, created_at, duration). Every call
    -- cache hit or live -- also appends one line to a JSONL log at
    `log_path` for easy aggregate analysis.

    On failure, a call is retried up to `max_retries` times with
    `retry_delay` seconds between attempts. If all attempts fail, the
    failure is logged and an empty string is returned.
    """

    def __init__(
        self,
        cache_dir: str,
        log_path: str = None,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        max_retries: int = 3,
        retry_delay: int = 10,
    ):
        self.cache_dir = cache_dir
        self.log_path = log_path or os.path.join(cache_dir, "calls.jsonl")
        self.model = model or os.environ.get("IA_ROUTER_MODEL", "UnB-Llama-3.3-70B-Instruct")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("IA_ROUTER_API_KEY"),
            base_url=base_url or os.environ.get("IA_ROUTER_BASE_URL"),
        )

    def _cache_key(self, system_prompt: str, prompt: str) -> str:
        return hashlib.sha1((system_prompt + prompt).encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def _append_log(self, entry: dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def run(self, prompt: str, system_prompt: str = "", temperature: float = 0.0, model: str = None, force: bool = False) -> str:
        """Returns just the response text. Use `run_with_meta` for token counts, timing, etc."""
        return self.run_with_meta(prompt, system_prompt=system_prompt, temperature=temperature, model=model, force=force)["response"]

    def run_with_meta(self, prompt: str, system_prompt: str = "", temperature: float = 0.0, model: str = None, force: bool = False) -> dict:
        """Same as `run`, but returns the full call record (response, token counts, raw_response, created_at, duration_seconds, cached).

        `force=True` skips a cache hit and makes a live call instead, overwriting the
        cache entry with the fresh result. Use this to redo a call whose cached response
        parsed as invalid/incomplete, since replaying the same prompt from cache would
        otherwise just return the same bad response forever.
        """
        model = model or self.model
        key = self._cache_key(system_prompt, prompt)
        cache_path = self._cache_path(key)

        if os.path.exists(cache_path) and not force:
            with open(cache_path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            entry = {**entry, "cached": True}
            self._append_log(entry)
            return entry

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exception = None
        created_at = None
        duration_seconds = None

        for attempt in range(1, self.max_retries + 1):
            created_at = datetime.now().isoformat()
            start = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                duration_seconds = time.monotonic() - start
                response_text = response.choices[0].message.content
                usage = response.usage

                entry = {
                    "prompt_hash": key,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "model": model,
                    "input_tokens": usage.prompt_tokens if usage else None,
                    "output_tokens": usage.completion_tokens if usage else None,
                    "response": response_text,
                    "raw_response": response.model_dump(),
                    "created_at": created_at,
                    "duration_seconds": duration_seconds,
                    "cached": False,
                }
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)

                self._append_log(entry)
                return entry

            except Exception as e:
                last_exception = e
                duration_seconds = time.monotonic() - start
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] [LLMRunner] "
                    f"attempt {attempt}/{self.max_retries} failed: {e}",
                    flush=True,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        failure_entry = {
            "prompt_hash": key,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "model": model,
            "input_tokens": None,
            "output_tokens": None,
            "response": "",
            "raw_response": None,
            "error": str(last_exception),
            "created_at": created_at,
            "duration_seconds": duration_seconds,
            "cached": False,
        }
        self._append_log(failure_entry)
        return failure_entry
