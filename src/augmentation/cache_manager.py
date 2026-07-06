import json
import os
import hashlib
from typing import List, Optional

class AugmentationCache:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2)

    def _generate_key(self, method: str, folder_id: str, doc_ids: Optional[List[str]] = None) -> str:
        """
        Generates a deterministic cache key.
        If doc_ids is None (empty folder), it's just method + folder_id.
        If doc_ids is provided, it sorts them and hashes them to ensure order independence.
        """
        base = f"{method}_{folder_id}"
        if doc_ids is None or len(doc_ids) == 0:
            return f"{base}_nodoc"
        
        sorted_docs = sorted(doc_ids)
        doc_hash = hashlib.md5(",".join(sorted_docs).encode()).hexdigest()
        return f"{base}_{doc_hash}"

    def get(self, method: str, folder_id: str, doc_ids: Optional[List[str]] = None):
        key = self._generate_key(method, folder_id, doc_ids)
        return self.cache.get(key)

    def set(self, method: str, folder_id: str, doc_ids: Optional[List[str]], result: dict):
        key = self._generate_key(method, folder_id, doc_ids)
        self.cache[key] = result
        self._save_cache()
