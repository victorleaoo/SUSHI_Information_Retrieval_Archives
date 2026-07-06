import hashlib
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, cache_dir: str, model: str = 'llama-3.3-70b-versatile', provider: str = 'groq'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.model = model
        self.provider = provider
        if provider == 'openai':
            from openai import OpenAI
            self.client = OpenAI()
        elif provider == 'groq':
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                print("Warning: GROQ_API_KEY environment variable not set.")
            self.client = Groq(api_key=api_key)

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

    def generate(self, prompt: str, system_prompt: str = '', temperature: float = 0.3) -> str:
        key = self._cache_key(system_prompt + prompt)
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f).get('response', '')
        
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.provider == 'openai':
                    response = self.client.chat.completions.create(
                        model=self.model, messages=messages, temperature=temperature
                    )
                elif self.provider == 'groq':
                    response = self.client.chat.completions.create(
                        model=self.model, messages=messages, temperature=temperature
                    )
                text = response.choices[0].message.content
                
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({'prompt_hash': key, 'response': text, 'model': self.model}, f, indent=2)
                
                # Sleep slightly to respect rate limits on groq if needed
                time.sleep(22)
                return text
                
            except Exception as e:
                print(f"Error calling LLM (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise e
