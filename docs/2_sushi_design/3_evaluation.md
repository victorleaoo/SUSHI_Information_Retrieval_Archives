# SUSHI Evaluation

This section is about the metrics to evaluate a system's performance.

## NDCG@5

The **principal evaluation measure** if a system construct the ranking of folders related to a search is Normalized Discounted Cumulative Gain at or above the 5th rank position (NDCG@5). The choice behind the number 5[1]:
- A cutoff at 5 corresponds to about a **half hour’s work by someone** who is actually looking at physical documents in an archive[1];
- We estimate this from the fact that a folder contains an average of **31682/1337 ≈ 24 documents**, together with out expectation that a skilled searcher could recognize a relevant document in **15 seconds**[1];
- Obtaining the **boxes** that contain those folders might take **another hour or two**[1].

Its design reflects an assumption that a searcher has time to examine no more that five folders, and that the searcher prefers to see the most highly relevant folders first when working down from the top of the ranked list [2].

## MAP

Mean Average Precision (MAP) has no cutoff. Its design reflects an assumption that the searcher wishes to find some (unknown) number of relevant folders, and that they prefer to achieve this while looking at the smallest number of folders when working down from the top of the ranked list [2].

## S@1
Success at 1 (S@1) assumes that the searcher will examine only the top-ranked folder. If it is relevant they will be fully satisfied. If not, they will be completely dissatisfied[2].

## Relevance Judgement File Format

Relevance judgment (QRels) files for the dry run topics and the official run topics will be distributed in the TSV format.
- **TopicID**: A topic identifier;
- **ZeroColumn**: The value for this field is always zero;
- **SushiFolder**: The Sushi Folder identifier for which a relevance judgment was made;
- **RelevanceDegree**: A nonnegative integer that indicates the graded relevance judgement for every judged document. Unjudged documents are omitted from the QRel file.
    * **3**: Highly Relevant;
    * **1**: Somewhat Relevant;
    * **0**: Not revelant.

```
T18-00001 0 N23812924 3
T18-00002 0 G99990322 1
T18-00003 0 N23812940 1
T18-00003 0 G99990184 0
```

## Future Work

The nDCG@5 simplifies the goal. Some level of complexity might be more interesting[1]:

- Approach using **density of relevant documents in a folder**:
    * In our present approach, **systems get no more credit for finding a folder with five relevant documents than for finding a folder with just one**.
- A **cost model** based on the discovery rate:
    * Evaluate the **time required to rank**.
- In the U.S. National Archives, for example, searchers **request access not to folders, but to the boxes** that contain the folders they want to see.
    * Prefer to find highly ranked folders that happen to be in the same box (or in nearby boxes).

## References

> 1. [Searching Unseen Sources for Historical Information: Evaluation Design for the NTCIR-18 SUSHI Pilot Task.](https://terpconnect.umd.edu/~oard/pdf/emtcir24.pdf)

> 2. [NTCIR-18 SUSHI Pilot Task Overview](https://drive.google.com/file/d/12P2g-A11nRW9CwFA7MFmA4jGvSoUXjNn/view?usp=sharing)