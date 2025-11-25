
# Proposed Work based on SUSHI Hard Topics and Explainable IR

Experiments conducted in Archival Information Retrieval to date consistently highlight a systemic failure in retrieval performance across a specific subset of topics (the "hard topics"), signaling a crucial gap in current approaches. Consequently, this work implements a diagnostic methodology designed to investigate and explain the underlying reasons why state-of-the-art systems fail to identify relevant documents for these challenging queries.

## Reproducibility and Benchmarking

Reproduce best models

## Application of ExIR Frameworks

Following the reproduction of the SUSHI task results and the establishment of baselines, the next critical phase involves the application of Explainable Information Retrieval (ExIR) techniques.

This stage applies a diverse set of model-agnostic explainability methods—ranging from pointwise to listwise approaches—to the ranked lists of folders generated for each topic. The objective is to obtain **multi-faceted interpretations of the ranking decisions to diagnose** the systemic failures observed in the “hard topics”.

- For **every topic** in the test set, the **ranked lists produced** by the target models (e.g., UMCP-TOFS variants) are aggregated.
- Subsequently, **specific ExIR models are applied** to these lists to **generate local explanations**.
    * **LIRME**: Explaning Vector with weights indicating contribution of specific terms to the final score;
    * **Rank-LIME**: Feature weights quantifying the relative contribution of each term to the ranking;
    * **The Curious Case of IR Explainability**: Coefficients that indicate the behavior for TF, DF/IDF and Length;
    * **MULTIPLEX**: <MELHOR ESTUDO>;
    * **Counterfactual**: set of missing terms that, if added to the document, could improve its rank.

Each of these frameworks operates on distinct premises, providing a comprehensive diagnostic toolkit.

## Failure Analysis and Synthesis

Having generated local explanation, the objective is to construct a comprehensive *Taxonomy of Failure Modes* that **explains the mechanical and semantic reasons behind the systemic underperformance** of state-of-the art models on the “hard topics” of the SUSHI task.

### Analytical Protocol

#### Diagnosis of False Positives (Why was the wrong folder retrieved?)

-  For “hard topics” where the system filled the top-5 ranks with irrelevant folders, we will utilize feature attribution explanations.
    * Did the model over-rely on a polysemous term in the folder label that matched the query lexically but not semantically?
    *  Did OCR artifacts (e.g., misrecognized headers or repetitive noise) falsely trigger the relevance scoring mechanism in neural models?

#### Diagnosis of False Negatives (Why was the relevant folder missed?)

- For relevant folders that failed to appear in the top ranks, we will utilize counterfactual explanations (from CFIR). Missing elements.
    * What terms did the counterfactual generator suggest adding to the document to boost its rank?
    * For instance:
        - Do these missing terms reveal a vocabulary mismatch (e.g., the query uses modern terminology while the document uses historical jargon)? 
        - Or do they indicate that the model requires a density of evidence (term frequency) that sparse archival records simply do not possess?

### Synthesis: A Taxonomy of Failure Mode

Instead of reporting isolated errors, this work aims to categorize them into structural classes, such as:

- **Lexical-Semantic Gap**: Cases where the **semantic bridge between the user’s query and the archival metadata is broken**;
- **Sensitivity to Sparse Evidence**: Cases where the model correctly identified a signal, but the signal’s strength (e.g., a single mention in a folder label) was insufficient to outweigh the noise from the document body in the ranking function;
- **Artifact-Induced Hallucination**: Instances where the specific characteristics of the digitization process (OCR errors, layout artifacts) actively misled the retrieval model.

## Strategic Recommendations for Future Research

Based on the final results of the previous stages, this work will propose evidence-based strategies for the next researches of archival search systems. Rather than generic suggestions, these recommendations will be directly mapped to the identified failure modes