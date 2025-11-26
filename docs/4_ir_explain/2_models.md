# Methods that can be used for ExIR

Below there are few models I choose to help perform the explainability for the SUSHI models.

## [LIRME](https://www.researchgate.net/publication/334582233_LIRME_Locally_Interpretable_Ranking_Model_Explanation)

### Fundamentals

- Introduces a **post-hoc, model-agnostic methodology**
- Why model assigns a specific relevance score to a **query-document pair (Q,D)**;
- Although modern ranking models often function as black boxes with complex internal term-weighting variations, their local behavior can be approximated. 
- This is achieved by estimating an explanation vector, where each dimension represents a term from the document. 
    * The weights within this vector indicate the contribution, **positive (for relevant terms) or negative**, of each specific term to the final document score.

### Techniques

- Learns the explanation vector by **analyzing how the black-box model’s score fluctuates** in response to **variations** of the original document;
- Identifies **not only the contribution of exact query terms but also highlights non-query terms** that are semantically related and contribute to the relevance score.

## [Rank-LIME](https://arxiv.org/abs/2212.12722)

### Fundamentals

- Proposes a **post-hoc, model-agnostic framework**;
- Existing **pointwise explainability methods are insufficient** because they fail to capture the rationale behind the relative ordering of a list.
- A **listwise approach is necessary to identify the specific features (terms)** responsible for positioning one document above another in a ranked list.

### Techniques

- Extends the standard LIME framework;
    * Instead of random perturbations, it employs **correlation-based perturbations**.
    * Replaces the standard regression loss function (Mean Squared Error) used in LIME with **differentiable ranking loss functions, such as ApproxNDCG or NeuralNDCG**.
- The final explanation delivered by Rank-LIME consists of a set of **linear feature weights** (additive feature attributions).
    * These weights **quantify the relative contribution of each feature** (e.g., a word in the query or document) towards the observed ordering of the results.

## [The Curious Case of IR Explainability: Explaining Document Scores within and across Ranking Models](https://gdebasis.github.io/files/RegressionBasedExplanation.pdf)

### Fundamentals

- The central premise **diverges from explaining decisions at the level of individual terms (as in LIME)**; 
- Instead, it seeks to explain the behavior of any ranking model in terms of its implicit weighting of the three fundamental building blocks of IR:
    * **term frequency in the document (TF)**,
    * **term frequency in the collection (DF/IDF)**,
    * **document length (Length)**. 
- Representing any model as a point in this "functional space" of dimensions (plus a fourth dimension, semantic similarity, for neural models), it becomes possible to directly compare distinct models and explain ranking differences both within a single model (intra-model) and across different models (inter-model).

### Techniques

- The method approximates the black-box similarity function of a model (M) using a **linear regression framework**;
- The resulting **regression coefficients θ = (θx,θy,θz,[θω])** constitute the “explanation vector” of model M, quantifying the **magnitude (importance)** and **direction (sign) of the influence each fundamental component** has on the final score.

## [Listwise Explanations for Ranking Models Using Multiple Explainers](https://research.tudelft.nl/files/150434358/978_3_031_28244_7_41.pdf)

=> Estudar mais para explicar aqui

## [A Counterfactual Explanation Framework for Retrieval Models](https://arxiv.org/abs/2409.00860)

### Fundamentals

- Unlike traditional methods that ask "Why is this document relevant?", CFIR addresses the negative outcome: **"Why was this document not favored (i.e., not retrieved in the top-K)?"**;
- **Counterfactual explanation**: identifying the minimal set of terms that, if added to the document, would cause it to rise in the ranking;

### Techniques

- The ranking task is cast as a **local binary classification problem**. 
    * Documents retrieved in the top-K of the black-box model are labeled as **Class 1 (favored)**.
    * The target document to be explained, along with its K nearest neighbors outside the top-K, are labeled as **Class 0**.
-  To handle the high dimensionality of text, a **local vocabulary V is constructed by aggregating the top-n most significant words** (extracted via BERT-Similarity) from the Class 1 documents.
- A simple surrogate classifier f (e.g., Random Forest) is **trained to distinguish Class 1 from Class 0**. Then, an optimization algorithm searches for a counterfactual vector c that minimizes a multi-objective loss function.
- The final explanation delivered is the set of terms (and their respective counts) found in the **difference between the counterfactual vector c and the original document vector d**.