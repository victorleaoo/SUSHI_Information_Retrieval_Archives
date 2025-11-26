# Overview of Information Retrieval Solutions

## [BM-25](https://www.researchgate.net/publication/220613776_The_Probabilistic_Relevance_Framework_BM25_and_Beyond)

[BM25](https://vishwasg.dev/blog/2025/01/20/bm25-explained-a-better-ranking-algorithm-than-tf-idf/) is a ranking function used by search engines to **estimate the relevance of documents to a given search query**. It is an **evolution of the classic TF-IDF** algorithm and is widely considered the state-of-the-art for sparse retrieval.

## Core Concepts

BM25 improves upon TF-IDF by introducing two main components:

1.  **Term Frequency (TF) Saturation:** In standard TF, relevance increases linearly with term frequency. In BM25, the score impact of a term appearing 100 times is not 10 times greater than if it appeared 10 times.
2.  **Document Length Normalization:** It penalizes long documents. A keyword appearing once in a short tweet is likely more significant than the same keyword appearing once in a 300-page book.

## Practical Example

Imagine a collection of 3 documents. We want to search for the query: **"cat"**.

**The Collection:**

* **Doc 1:** "The cat sits." (Length: 3)
* **Doc 2:** "The cat chases the other cat." (Length: 6)
* **Doc 3:** "The dog barks." (Length: 3)

**Constants:**

* avgdl = (3 + 6 + 3) / 3 = 4
* k1 = 1.2
* b = 0.75

### Step 1: Calculate IDF for "cat"

"cat" appears in 2 out of 3 docs.
IDF ≈ ln( (N - n + 0.5) / (n + 0.5) + 1 )
IDF("cat") ≈ 0.47

### Step 2: Calculate Score for Doc 1 ("The cat sits")

* f("cat", D1) = 1
* |D1| = 3

Score = 0.47 * [ (1 * 2.2) / (1 + 1.2 * (1 - 0.75 + 0.75 * (3/4))) ]
Score ≈ **0.52**

### Step 3: Calculate Score for Doc 2 ("The cat chases the other cat")

* f("cat", D2) = 2
* |D2| = 6

Score = 0.47 * [ (2 * 2.2) / (2 + 1.2 * (1 - 0.75 + 0.75 * (6/4))) ]
Score ≈ **0.56**

### Result Analysis

* **Doc 2 (0.56)** ranks higher than **Doc 1 (0.52)**.
* Even though Doc 2 is twice as long (which penalizes the score), the fact that "cat" appears twice overcame the length penalty. However, notice the scores are close; if Doc 2 were much longer with only 2 mentions, Doc 1 would win.