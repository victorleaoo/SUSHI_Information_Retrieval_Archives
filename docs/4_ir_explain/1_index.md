# Explainable Information Retrieval (ExIR)

The hard topics problem appoint to the need of the application of Explainable Information Retrieval (ExIR). Traditional performance metrics quantify the extent of the failure but are blind to its mechanical causes. Therefore, a specific study employing explainability techniques is not merely an addition but a necessity to dissect the rankers. The objective is to transition from speculative hypotheses to evidence-based insights, explaining how these techniques interact with archival data to cause the "hard topics"phenomenon.

## Concepts

In the context of machine learning, interpretability is defined as "the ability to explain or to present in understandable terms to a human".

In ranking, the objective is not merely to assign a label, but to understand why a document is considered relevant to an arbitrary query. Furthermore, since retrieval models encode a query intent that drives the output ranking, explainability methods should elucidate what the model understands when a specific query is issued.

Explainable Information Retrieval (ExIR) methods are also distinguished by the granularity of the elements they explain.
- **Pointwise** methods focus on explaining the relevance score or decision for a single document within that list;
- **Pairwise** methods aim to explain the model’s preference for one document over another (e.g., why document A is ranked higher than document B).
- **Listwise** methods view the ranking task as an aggregation of multiple pointwise or pairwise decisions and aim to cover the rationale behind the entire ranked list.

Interpretability approaches are classified based on their relationship with the model architecture.
- **Post-hoc** interpretability methods explain the decisions of already trained models, seeking primarily to elucidate the output rather than the inner workings.
- **Interpretable-by-Design (IBD)** models, which are constructed to be inherently transparent, allowing predictions to be unambiguously attributed to specific parts of the input.