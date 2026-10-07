import os
import dspy
from dotenv import load_dotenv
from dspy.lm15 import CacheConfig
import dspy._vendor.lm15.providers.openai as _oa
from src.dspy_experiments.common import LUNA_ROOT

load_dotenv()

# DSPy 3.4.0 only sends OpenAI's "explicit" cache mode to models named gpt-X.Y;
# gpt-6-luna doesn't match, so every prompt was written to the cache at 1.25x input price.
_orig = _oa.openai_model_has_cache_options
_oa.openai_model_has_cache_options = lambda m: m.lower().startswith("gpt-6") or _orig(m)

PROMPT_CACHE = CacheConfig(prefix="stable", key="sushi-luna")  # cache only system msg + instructions

dspy.configure_cache(enable_disk_cache=True,
                     disk_cache_dir=os.path.join(LUNA_ROOT, "dspy_cache"))
TASK_LM = dspy.LM("openai/gpt-6-luna", reasoning_effort="none", num_retries=8, prompt_cache=PROMPT_CACHE)
REFLECTION_LM = dspy.LM("openai/gpt-6-luna", reasoning_effort="high",
                        service_tier="flex", prompt_cache=PROMPT_CACHE)
dspy.configure(lm=TASK_LM, adapter=dspy.JSONAdapter(), track_usage=True)