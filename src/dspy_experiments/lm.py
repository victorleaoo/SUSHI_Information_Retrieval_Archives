import logging
import os

import dspy
from dotenv import load_dotenv

from src.dspy_experiments.common import LUNA_ROOT

logging.getLogger("dspy.predict.predict").setLevel(logging.ERROR)

load_dotenv()
dspy.configure_cache(enable_disk_cache=True,
                     disk_cache_dir=os.path.join(LUNA_ROOT, "dspy_cache"))   # absolute path

TASK_LM = dspy.LM("openai/gpt-6-luna", reasoning_effort="none", num_retries=8)
REFLECTION_LM = dspy.LM("openai/gpt-6-luna", reasoning_effort="high")

dspy.configure(lm=TASK_LM, adapter=dspy.JSONAdapter(), track_usage=True)