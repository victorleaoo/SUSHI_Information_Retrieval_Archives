# Phase 1: All Folders Index Experiments Analysis

This report provides a tabular comparison of all runs from Phase 1. It helps visually compare 1A runs amongst themselves, 1B runs amongst themselves, and against each other.

## Phase 1A: Base Queries (Q0) across Configurations
The table below shows **nDCG@5** for different index configurations (F1-F5) across all retrieval models.

| Model   |     F1 |     F2 |     F3 |     F4 |     F5 |
|:--------|-------:|-------:|-------:|-------:|-------:|
| B       | 0.1268 | 0.1195 | 0.1322 | 0.1201 | 0.1204 |
| E       | 0.1032 | 0.1068 | 0.1235 | 0.0907 | 0.0934 |
| C       | 0.1049 | 0.1111 | 0.0981 | 0.1104 | 0.1055 |
| BC      | 0.1125 | 0.1189 | 0.0927 | 0.1051 | 0.1019 |
| BE      | 0.1181 | 0.1083 | 0.1076 | 0.1083 | 0.1106 |
| CE      | 0.1068 | 0.1222 | 0.1035 | 0.1154 | 0.1201 |
| BCE     | 0.1143 | 0.1249 | 0.1008 | 0.1060 | 0.1184 |

### Best Configuration per Model (Baseline for 1B)
| Model   | Config   |   nDCG@5 |
|:--------|:---------|---------:|
| B       | F3       |   0.1322 |
| BCE     | F2       |   0.1249 |
| E       | F3       |   0.1235 |
| CE      | F2       |   0.1222 |
| BC      | F2       |   0.1189 |
| BE      | F1       |   0.1181 |
| C       | F2       |   0.1111 |

## Phase 1B (1-4): LLM Expansions Alone
These runs replace Q0 entirely with LLM-generated text.

| Model   |   Q0 (Baseline) |     Q1 |       Q2 |       Q3 |     Q4 |     QALL |
|:--------|----------------:|-------:|---------:|---------:|-------:|---------:|
| B       |          0.1322 | 0.1062 |   0.0292 | nan      | 0.0865 | nan      |
| E       |          0.1235 | 0.0757 | nan      |   0.0048 | 0.0547 | nan      |
| C       |          0.1111 | 0.0513 | nan      | nan      | 0.0071 | nan      |
| BC      |          0.1189 | 0.0962 |   0.0000 | nan      | 0.0412 | nan      |
| BE      |          0.1181 | 0.1097 | nan      |   0.0761 | 0.0618 | nan      |
| CE      |          0.1222 | 0.0978 | nan      |   0.0000 | 0.0538 | nan      |
| BCE     |          0.1249 | 0.0980 |   0.0067 |   0.0593 | 0.0414 |   0.0809 |

## Phase 1B (5-6): Q0-Augmented Queries
These runs preserve the original query (Q0) and augment it with LLM-generated text.

| Model   |   Q0 (Baseline) |    Q0+Q1 |    Q0+Q2 |    Q0+Q3 |    Q0+Q4 |   Q0+QALL |
|:--------|----------------:|---------:|---------:|---------:|---------:|----------:|
| B       |          0.1322 |   0.1136 |   0.0736 | nan      |   0.1325 |    0.1160 |
| E       |          0.1235 |   0.0829 |   0.0109 |   0.0273 |   0.0760 |    0.0395 |
| C       |          0.1111 | nan      | nan      | nan      |   0.0513 |  nan      |
| BC      |          0.1189 |   0.0835 | nan      | nan      | nan      |  nan      |
| BE      |          0.1181 | nan      | nan      |   0.0898 | nan      |  nan      |
| CE      |          0.1222 | nan      |   0.0734 | nan      | nan      |  nan      |
| BCE     |          0.1249 |   0.1342 |   0.1231 |   0.0984 |   0.0918 |    0.0987 |

## Phase 1B (7): Query-Index Interactions (Q0+QALL × Configs)
Tests whether richer index metadata interacts with the comprehensive augmented query.

| RunID   | Config   |   nDCG@5 |
|:--------|:---------|---------:|
| 1B-36   | F1       |   0.1061 |
| 1B-37   | F2       |   0.0987 |
| 1B-38   | F3       |   0.0992 |
| 1B-39   | F4       |   0.1029 |
| 1B-40   | F5       |   0.0988 |

## 🏆 Overall Top 15 Configurations (1A + 1B combined)
|   Rank | Model   | QueryType   | Config   |   nDCG@5 |
|-------:|:--------|:------------|:---------|---------:|
|      1 | BCE     | Q0+Q1       | F2       |   0.1342 |
|      2 | B       | Q0+Q4       | F3       |   0.1325 |
|      3 | B       | Q0          | F3       |   0.1322 |
|      4 | B       | Q0          | F1       |   0.1268 |
|      5 | BCE     | Q0          | F2       |   0.1249 |
|      6 | E       | Q0          | F3       |   0.1235 |
|      7 | BCE     | Q0+Q2       | F2       |   0.1231 |
|      8 | CE      | Q0          | F2       |   0.1222 |
|      9 | B       | Q0          | F5       |   0.1204 |
|     10 | B       | Q0          | F4       |   0.1201 |
|     11 | CE      | Q0          | F5       |   0.1201 |
|     12 | B       | Q0          | F2       |   0.1195 |
|     13 | BC      | Q0          | F2       |   0.1189 |
|     14 | BCE     | Q0          | F5       |   0.1184 |
|     15 | BE      | Q0          | F1       |   0.1181 |

