import dspy
from src.dspy_experiments.lm import TASK_LM
from src.dspy_experiments.signatures import CoreThemes, CTX

ct = dspy.Predict(CoreThemes)
pred = ct(collection_context=CTX,
          information_need="Brazilian submarines\nFind documents that mention the "
                           "operation of submarines by the Brazilian Navy.")
print(pred.core_themes, pred.related_concepts)
print(pred.get_lm_usage())          # look for completion_tokens_details.reasoning_tokens
dspy.inspect_history(n=1)           # the exact request sent, for your records