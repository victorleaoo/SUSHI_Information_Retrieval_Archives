import hashlib
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """
    Unified LLM client supporting Groq, OpenAI, and Ollama (local) providers.

    Features:
    - SHA256-based disk caching: each prompt is cached so repeated calls are free.
    - Structured logging: every call (cached or live) optionally writes a detailed
      JSON log entry to `logs_dir` for traceability.

    Args:
        cache_dir (str): Directory for cached responses.
        model (str): Model identifier. Defaults differ per provider:
            - groq: 'llama-3.3-70b-versatile'
            - ollama: 'qwen3:12b'
            - openai: 'gpt-4o'
        provider (str): One of 'groq', 'openai', 'ollama'.
        logs_dir (str | None): If set, writes a JSON log per call to this directory.
        ollama_url (str): Base URL for local Ollama server.
    """
    def __init__(
        self,
        cache_dir: str,
        model: str = None,
        provider: str = 'ollama',
        logs_dir: str = None,
        ollama_url: str = 'http://localhost:11434',
    ):
        self.cache_dir = cache_dir
        self.provider = provider
        self.logs_dir = logs_dir
        self.ollama_url = ollama_url
        os.makedirs(cache_dir, exist_ok=True)
        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)

        # Set default model per provider
        if model is None:
            _defaults = {'groq': 'llama-3.3-70b-versatile', 'openai': 'gpt-4o', 'ollama': 'qwen3:12b'}
            model = _defaults.get(provider, 'qwen3:12b')
        self.model = model

        if provider == 'openai':
            from openai import OpenAI
            self.client = OpenAI()
        elif provider == 'groq':
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                print("Warning: GROQ_API_KEY environment variable not set.")
            self.client = Groq(api_key=api_key)
        elif provider == 'ollama':
            import requests
            self._requests = requests  # store for use in generate()
            # Quick connectivity check
            try:
                r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                r.raise_for_status()
                print(f"[LLMClient] Connected to Ollama at {self.ollama_url} (model: {self.model})")
            except Exception as e:
                print(f"[LLMClient] Warning: Could not reach Ollama at {self.ollama_url}: {e}")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'groq', or 'ollama'.")

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

    def _write_log(self, log_entry: dict):
        """Writes a structured JSON log entry to logs_dir."""
        if not self.logs_dir:
            return
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        hash_prefix = log_entry.get('prompt_hash', 'unknown')[:8]
        log_filename = f"{timestamp_str}_{hash_prefix}.json"
        log_path = os.path.join(self.logs_dir, log_filename)
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)

    def generate(self, prompt: str, system_prompt: str = '', temperature: float = 0.3) -> str:
        """
        Generates a response for the given prompt, using cache if available.

        Args:
            prompt (str): The user message / prompt.
            system_prompt (str): Optional system message.
            temperature (float): Sampling temperature (ignored for cached results).

        Returns:
            str: The model's text response.
        """
        key = self._cache_key(system_prompt + prompt)
        cache_path = os.path.join(self.cache_dir, f"{key}.json")

        # --- Cache hit ---
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            text = cached.get('response', '')
            self._write_log({
                'timestamp': datetime.now().isoformat(),
                'cached': True,
                'provider': self.provider,
                'model': self.model,
                'prompt_hash': key,
                'system_prompt': system_prompt,
                'prompt': prompt,
                'response': text,
            })
            return text

        # --- Build messages ---
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        # --- Live call with retries ---
        max_retries = 3
        text = ''
        for attempt in range(max_retries):
            try:
                if self.provider in ('openai', 'groq'):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                    )
                    text = response.choices[0].message.content

                elif self.provider == 'ollama':
                    resp = self._requests.post(
                        f"{self.ollama_url}/api/chat",
                        json={
                            'model': self.model,
                            'messages': messages,
                            'stream': False,
                            'options': {'temperature': temperature},
                        },
                        timeout=300,
                    )
                    resp.raise_for_status()
                    text = resp.json()['message']['content']

                # --- Cache write ---
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({'prompt_hash': key, 'response': text, 'model': self.model}, f, indent=2, ensure_ascii=False)

                # Rate-limit courtesy sleep for Groq
                if self.provider == 'groq':
                    time.sleep(22)

                break  # success

            except Exception as e:
                print(f"[LLMClient] Error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise

        # --- Log entry ---
        self._write_log({
            'timestamp': datetime.now().isoformat(),
            'cached': False,
            'provider': self.provider,
            'model': self.model,
            'prompt_hash': key,
            'system_prompt': system_prompt,
            'prompt': prompt,
            'response': text,
        })

        return text

