## SUSHI Experiment Running

This system is designed to simulate Information Retrieval scenarios on sparsely digitized archival collections (SUSHI). It runs experiments by sampling subsets of the collection (using random seeds), training models on those subsets, expanding results for folders that have no documents in the ECF, and evaluating the performance.

### 1. System/File Architecture

The pipeline workflow is built on four modular classes, each with a distinct responsibility.

- **1. RunGenerator (*src/run_generator.py*)**
    * **Role:** The main controller. It manages the experiment loop, initializes models, coordinates data flow, and executes the retrieval pipeline.
    * **Responsibility:** It delegates tasks to the helper classes below. It creates the "Experiment Collection Format" (ECF) which defines which documents are "visible" (digitized) for a specific run.

- **2. DataLoader (*src/data_loader.py*)**
    * **Role:** Manages file I/O and data structure.
    * **Responsibility:**
        * Loads metadata (`FoldersV1.3.json`, `itemsV1.2.json`).
        * Maps the physical directory structure (`Box -> Folder -> File`).
        * Generates the **ECF (Experimental Collection Format)**. This involves randomly sampling documents per box based on a specific random seed to create a training set.

- **3. RetrievalModel (*src/models.py*)**
    * **Role:** Abstract base class for search algorithms.
    * **Subclasses:**
        * `BM25Model`: Uses **PyTerrier** for sparse, frequency-based retrieval. Can automatically switch between BM25 and BM25F (field-weighted) based on input.
        * `EmbeddingsModel`: Uses **SentenceTransformers** (e.g., all-mpnet-base-v2) for dense vector retrieval via Cosine Similarity.
        * `ColBERTModel`: Uses **PyLate/ColBERT** for late-interaction retrieval (token-to-token matching) using PLAID indexing.
    * **Standard:** Every model must implement `train(data)` to build an index and `search(query)` to return a DataFrame of results.

- **4. Evaluator (*src/evaluator.py*)**
    * **Role:** Computes performance metrics.
    * **Responsibility:**
        * Converts model outputs into standard **TREC Run Files**.
        * Compares results against QRELs (Ground Truth) using `pytrec_eval`.
        * Calculates **nDCG@5**, **MAP**, and a custom metric: **Top-5 Relevant Folder Count**.
        * Aggregates results across all random seeds to produce Mean scores and 95% Confidence Intervals.

### 2. The Experiment Workflow (Step-by-Step)

By executing `RunGenerator.run_experiments()`, the system follows this lifecycle:

**Phase A: Setup**

1.  **Configuration:** The system reads the parameters (fields to index, models to use, expansion techniques).
2.  **Directory Prep:** Creates a unique output folder name based on the config (e.g., `F_SB-SS_TD_BM25-COLBERT`).

**Phase B: The Simulation Loop (Per Random Seed)**

For `run_type='random'`, the system iterates through 30 fixed random seeds. For each seed:

1.  **Sampling (ECF Creation):**
    * The `DataLoader` selects 5 random documents from every box in the collection.
    * These documents become the "Training Set." All other documents are considered "undigitized" and not described by document-level metadata, and invisible to the model training.
2.  **Model Training:**
    * The active models (`BM25`, `ColBERT`, etc.) build their indexes **only** using the sampled Training Set.
3.  **Retrieval:**
    * The system iterates through the 45 standard Topics.
    * It constructs a query (e.g., Title + Description).
    * Each model searches its index and returns a ranked list of candidates.
4.  **Fusion & Expansion (The Complex Part):**
    * **RRF:** Results from multiple models (e.g., BM25 + ColBERT) are merged using Reciprocal Rank Fusion (RRF) at the *document* or *folder* level first.
    * **Expansion:** The system looks for "Ghost Folders" (folders not retrieved by the model). It checks the retrieved documents for relationships (Same Box, Same Classification Code). If enough evidence exists, the empty folder is assigned an inferred score.
    * **Safety Ceiling:** The score of inferred folders is mathematically capped so they cannot rank higher than the Top-K original results (controlled by `expansion_ceiling_k`).
5.  **Evaluation:**
    * The final ranked list of folders is saved.
    * Metrics (nDCG@5, Precision) are calculated for this specific seed.
    * For each topic, it saves how many relevant folders (qrels_value > 0) are in the top 5.

**Configuration Arguments**

The `RunGenerator` is highly configurable. Here is what each argument controls:

| Argument | Type | Description |
| :--- | :--- | :--- |
| `searching_fields` | `List[List[str]]` | Which metadata fields to index. <br>Ex: `[['title', 'ocr']]`. If multiple fields are provided, BM25 upgrades to BM25F automatically. |
| `query_fields` | `List[str]` | Which parts of the Topic to use as the query.<br>`'T'`: Title only.<br>`'TD'`: Title + Description.<br>`'TDN'`: Title + Desc + Narrative. |
| `run_type` | `str` | `'random'`: Runs the loop over 30 seeds with uniform 5 docs/box (simulating sparsity).<br>`'uneven'`: Runs the loop over 30 seeds with skewed distribution.<br>`'all_documents'`: Runs once using the entire collection (Oracle mode). |
| `models` | `List[str]` | The models to ensemble. Options: `'bm25'`, `'embeddings'`, `'colbert'`. If more than one is given, it is performed RRF between the different models results. |
| `expansion` | `List[str]` | Strategies to infer missing folders scores.<br>`'same_box'`: Neighbor is in the same box.<br>`'same_snc'`: Neighbor has same Classification Code.<br>`'close_date'`: Neighbor has same SNC and is temporally close.<br>`[]`: No expansion. |
| `rrf_input` | `str` | **`'docs'`**: Fuses model results at document level.<br>**`'folders'`**: Expands each model independently, then fuses final folders. |
| `expansion_ceiling_k` | `int` | **Trust Threshold**. Determines the rank `k` that expanded results cannot beat.<br>`1`: Expansion can take Rank #2 but not #1.<br>`2`: Expansion can take Rank #3 but not #2.<br>`3`: Expansion can take Rank #4, but Top 3 are preserved.<br>... |
| `all_folders_folder_label` | `bool` | If `True`, ignores document contents and retrieves based ONLY on folder metadata labels. The `searching_fields` must be only ['folderlabel'] |

### 3. Output Structure

After running an experiment, the `../all_runs/` directory will contain a folder named after your configuration (e.g., `TOFS_SB_TD_BM25`). Inside:

1.  **`Random{SEED}_TopicsFolderMetrics.json`**:
    * Detailed metrics for that specific seed run.
    * Contains `ndcg_cut_5` and `count_relevant_top5` for every topic.
2.  **`model_overall_stats.json`**:
    * Contains the **Global Mean nDCG@5** and the **Margin of Error** (95% CI) aggregated across all seeds.
3.  **`topics_mean_margin.json`**:
    * The Mean nDCG and Confidence Interval for *each specific topic* across all seeds.
4.  **`topics_relevant_count_stats.json`**:
    * The average number of relevant folders found in the Top 5 for each topic.

### 4. Usage Example

To run a hybrid experiment using **BM25 and ColBERT, searching Titles and OCR, using 'Same Box' expansion, and fusing results at the document level**:

1. **Install Python**: [https://www.python.org](https://www.python.org).
2. **Install Python libraries**: run the command ```pip install -r requirements.txt```. It is recommended to use a [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html) or a [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) env.
3. **Install Java for Pyterrier**: [https://pyterrier.readthedocs.io/en/latest/troubleshooting/java.html](https://pyterrier.readthedocs.io/en/latest/troubleshooting/java.html)
    - If you're running on Windows, change the ```java_home```, in the ```src/models.py``` file (line 112);
4. **Run the RunGenerator**: Change the parameters for the *src/run_generator.py* and run it (python3 run_generator.py) or create a new file, import the class and run:

```python
from run_generator import RunGenerator

# Configure the experiment
gen = RunGenerator(
    searching_fields=[['title', 'ocr']],
    query_fields=['TD'],
    run_type='random',
    models=['bm25', 'colbert'],
    expansion=['same_box'],
    rrf_input='docs',
    expansion_ceiling_k=3
)

# Execute
gen.run_experiments()
```

**IMPORTANT NOTE**: the code doesn't automatically delete the terrierindex folder that is created for each run, therefore, it is necessary to **manually delete it after the run generator stops** running.

### 5. Hybrid Models (Combining two different techniques with RRF) - `hybrid_models.py`

This script is an advanced tool designed to **fuse distinct retrieval strategies** into a single, optimized ranking. While the standard `RunGenerator` ensembles models that share the same configuration (e.g., BM25 + ColBERT both using the same document text), this script allows you to combine fundamentally different approaches.

**Key Use Case:** Combining **Document Retrieval** (using OCR content) with **Folder Retrieval** (using only folder metadata).

**1. How It Works**

The script defines two separate `RunGenerator` instances (`gen_A` and `gen_B`) and fuses their outputs using **Weighted Reciprocal Rank Fusion (RRF)**.

1.  **Run Config A (Content-Based):**
    * Typically uses rich document fields (`Title`, `OCR`, `Summary`).
    * Applies expansion techniques (e.g., `Same Box`) to infer folder relevance from document hits.
2.  **Run Config B (Metadata-Based):**
    * Uses **`all_folders_folder_label=True`**. This ignores file content and retrieves based purely on the folder's label.
3.  **Fusion (RRF):**
    * The results from A and B are merged.
    * You can assign weights (e.g., `1.0` for Content, `0.65` for Metadata) to prioritize one strategy over the other.

**2. Usage Guide**

To create your own hybrid experiment, open `src/hybrid_models.py` and modify the `run_hybrid_experiment` function.

Set up the two generators. Note how `gen_B` is set to `all_folders_folder_label=True`, making it a pure metadata run.

```python
# Configuration A: The "Deep Diver" (Document Content)
gen_A = RunGenerator(
    searching_fields=[['title', 'ocr']],
    models=['bm25', 'colbert'],
    expansion=['same_box'],
    rrf_input='docs',
    all_folders_folder_label=False  # <--- Standard Mode
)

# Configuration B: The "Overviewer" (Folder Metadata)
gen_B = RunGenerator(
    searching_fields=[['folderlabel']],
    models=['colbert'],
    expansion=[],
    all_folders_folder_label=True   # <--- Metadata Mode
)
```

**3. Name the Output**

Update the run_folder_name variable. This will be the directory created in all_runs/, so make it descriptive.

```Python
run_folder_name = "HYBRID-TOFS-SMS-1-ALLFL-COLBERT_NE_TD_BM25-EMBEDDINGS-COLBERT-TUNED-WRRF"
```

**4. Run the script directly from your terminal:**

```Bash
python src/hybrid_models.py
```

The results will be saved and evaluated automatically, ready for inspection in the Visualizer.

### 6. Wilcoxon Test Analysis Notebook

The [wilcoxon_test.ipynb](https://github.com/victorleaoo/SUSHI_Information_Retrieval_Archives/blob/main/src/stats_test/wilcoxon_test.ipynb) performs statistical significance testing to compare the performance of two models. Specifically, it uses the **Wilcoxon Signed-Rank Test** to evaluate whether the difference in performance metrics between two models is statistically significant across multiple random seed trials. 

It has 2 main type of analysis:

**1. Global Performance Comparison**

Computes the aggregate performance difference across all topics.

* **Outputs:**
    * **P-value:** Determines significance ($p < 0.05$).
    * **Win/Loss Count:** Shows how many seeds favored each model.
    * **Rank Breakdown:** A DataFrame detailing the score difference for each seed.

**2. Single Topic Deep Dive**

Provides a detailed analysis for a specific topic ID (e.g., `T18Eval-00001`).

* **Visualization:** Plots the score distribution across the 30 seeds for that specific query.
* **Robustness Check:** Verifies if the improvement on a specific topic is consistent or an outlier.

---