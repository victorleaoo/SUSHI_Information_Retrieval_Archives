import dspy
from dotenv import load_dotenv

load_dotenv()
dspy.configure_cache(enable_disk_cache=True,
                     disk_cache_dir="data/llm_calls_luna/dspy_cache")

TASK_LM = dspy.LM("openai/gpt-6-luna", reasoning_effort="none", num_retries=8)
REFLECTION_LM = dspy.LM("openai/gpt-6-luna", reasoning_effort="high")

dspy.configure(lm=TASK_LM, adapter=dspy.JSONAdapter(), track_usage=True)