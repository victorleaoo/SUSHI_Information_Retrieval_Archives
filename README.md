# SUSHI (Searching Unseen Sources for Historical Information) Experiment Runner and Visualizer

## Repository Setup

To run the applications correctly, it is necessary to reproduce the data structure locally, as large or sensitive files are not versioned in this repository (they are git-ignored).

```
├── all_runs/ # It contains runs that were already made from models  
├── data/                           
│   ├── folders_metadata/
│   │   └── FoldersV1.2.json
│   ├── items_metadata/
│   │   └── itemsV1.2.json          # ⚠️ Download Document metadata
│   ├── raw/                        # ⚠️ Download and Place the raw Box/Folder structure with PDFs here
│   │   └── A0001
│   │   └── A0002
│   │   └── A0003
│   │   └── ...
├── ecf/                            
├── qrels/                          
│   ├── dry-run-qrels/              
│   │   ├── Ntcir18SushiDryRunBoxQrelsV1.1.tsv
│   │   └── Ntcir18SushiDryRunFolderQrelsV1.1.tsv 
│   └── formal-run-qrels/          
│       ├── formal-box-qrel.txt
│       ├── formal-document-qrel.txt
│       └── formal-folder-qrel.txt
├── src/ # Experiment Run Generator
├── web_app/ # SUSHI visualizer
├── .gitignore
├── README.md
└── requirements.txt                # Python dependencies
```

### Adding missing files

All files can be found at the [SUSHI Test Collection](https://sites.google.com/view/ntcir-sushi-task/test-collection):

- [itemsV1.2.json](https://drive.google.com/file/d/1c_hpR_lgdGeskXaNQTdCS7nO9s1R2NOb/view?usp=share_link)
- [raw](https://drive.google.com/file/d/1hA5FW0cNloi20coLlGvnv5wMap8ZN8YL/view?usp=sharing)

After downloading, reproduce the following steps:

1. Create a dir called ```data```;
2. Inside the dir ```data/items_metadata```, place the ```itemsV1.2.json``` file;
3. Inside the dir ```data/raw```, place the ```sushi-files.zip``` inside it and unzip it. After unziping, bring all folders out to the ```raw``` dir and delete the ```sushi-files``` folder and the ```sushi-files.zip``` file.

### About Folders Metadata File

The **Folder Label Metadata** is a critical component of the retrieval pipeline. Since the raw folder names in archival collections are often opaque codes (e.g., `POL 15-1`), we must enrich them with semantic meaning derived from the **SNC** system used in [1963](https://www.archives.gov/files/research/foreign-policy/state-dept/finding-aids/records-classification-handbook-1963.pdf) and [1965](https://www.archives.gov/files/research/foreign-policy/state-dept/finding-aids/dos-records-classification-handbook-1965-1973.pdf).

This process is handled by the `FolderLabelConstructor` class (*src/data_creation/SNCLabelTranslate.py* file), which merges the raw folder metadata with an SNC Translation Table (`SncTranslationV1.3.xlsx`) to create a rich, searchable textual representation for every folder.

**1. Input Data Sources**

- **`SncTranslationV1.3.xlsx`**: A reference table containing the definitions for every SNC code. It has four key columns:
    * `SNC`: The code itself (e.g., `POL 1`).
    * `1965`: The label used in the 1965 classification (Preferred).
    * `1963`: The label used in the 1963 classification (Fallback).
    * `Scope Note`: A descriptive paragraph explaining what is included or excluded in this category.

**2. The Output Fields**

The final JSON output for each folder contains a set of enriched fields. Here is why each one exists:

| Field | Description | Why it exists? |
| :--- | :--- | :--- |
| `label1965` | The literal text from the 1965 column. | Preferred term source. |
| `label1963` | The literal text from the 1963 column. | Fallback source if 1965 is missing. |
| `main_title` | The resolved primary label (1965 or 1963). | Used as the core display name for the folder. |
| `raw_scope` | The full scope note. | Maximum context. |
| `scope_truncated` | A cleaned version. The code scans for "Stopper Keywords" (like `SEE`, `Exclude`, `Except`) and cuts off the text immediately before them. This leaves only the positive definition of the category. | High-precision context. Removes negative keywords that might confuse models. |
| `parent` | The SNC code of the immediate parent. | Used for field below. |
| `label_parent_expanded` | SNC codes are hierarchical. `POL 15-1` is a child of `POL 15`, which might be a child of `POL`. To understand `POL 15-1`, you need the context of its parents. | It provides the full semantic path, ensuring a folder like "General" is understood as "POLITICAL -> GENERAL". |

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
    * These documents become the "Training Set." All other documents are considered "undigitized" and invisible to the model training.
2.  **Model Training:**
    * The active models (`BM25`, `ColBERT`, etc.) build their indexes **only** using the sampled Training Set.
3.  **Retrieval:**
    * The system iterates through the 45 standard Topics.
    * It constructs a query (e.g., Title + Description).
    * Each model searches its index and returns a ranked list of candidates.
4.  **Fusion & Expansion (The Complex Part):**
    * **RRF:** Results from multiple models (e.g., BM25 + ColBERT) are merged using Reciprocal Rank Fusion (RRF) at the *document* or *folder* level first.
    * **Expansion:** The system looks for "Ghost Folders" (folders not retrieved by the model). It checks the retrieved documents for relationships (Same Box, Same Classification Code). If enough evidence exists, the empty folder is assigned an inferred score.
    * **Safety Ceiling:** Rhe score of inferred folders is mathematically capped so they cannot rank higher than the Top-K original results (controlled by `expansion_ceiling_k`).
5.  **Evaluation:**
    * The final ranked list of folders is saved.
    * Metrics (nDCG@5, Precision) are calculated for this specific seed.
    * For each topic, it saves how many relevant folders (qrels_value > 0) are on the top 5.

**Configuration Arguments**

The `RunGenerator` is highly configurable. Here is what each argument controls:

| Argument | Type | Description |
| :--- | :--- | :--- |
| `searching_fields` | `List[List[str]]` | Which metadata fields to index. <br>Ex: `[['title', 'ocr']]`. If multiple fields are provided, BM25 upgrades to BM25F automatically. |
| `query_fields` | `List[str]` | Which parts of the Topic to use as the query.<br>`'T'`: Title only.<br>`'TD'`: Title + Description.<br>`'TDN'`: Title + Desc + Narrative. |
| `run_type` | `str` | `'random'`: Runs the loop over 30 seeds (simulating sparsity).<br>`'all_documents'`: Runs once using the entire collection (Oracle mode). |
| `models` | `List[str]` | The models to ensemble. Options: `'bm25'`, `'embeddings'`, `'colbert'`. If more than one is given, it is performed RRF between the different models results. |
| `expansion` | `List[str]` | Strategies to infer missing folders scores.<br>`'same_box'`: Neighbor is in the same box.<br>`'same_snc'`: Neighbor has same Classification Code.<br>`'close_date'`: Neighbor has same SNC and is temporally close.<br>`[]`: No expansion. |
| `rrf_input` | `str` | **`'docs'`**: Fuses model results at document level.<br>**`'folders'`**: Expands each model independently, then fuses final folders. |
| `expansion_ceiling_k` | `int` | **Trust Threshold**. Determines the rank `k` that expanded results cannot beat.<br>`1`: Expansion can take Rank #2 but not #1.<br>`3`: Expansion can take Rank #4, but Top 3 are preserved. |
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

### 5. Hybrid Models (Combining two different techniques with RRF) - `rrf_best_models.py`

This script is an advanced tool designed to **fuse distinct retrieval strategies** into a single, optimized ranking. While the standard `RunGenerator` ensembles models that share the same configuration (e.g., BM25 + ColBERT both using the same document text), this script allows you to combine fundamentally different approaches.

**Key Use Case:** Combining high-precision **Document Retrieval** (using OCR content) with high-recall **Folder Retrieval** (using only folder metadata).

**1. How It Works**

The script defines two separate `RunGenerator` instances (`gen_A` and `gen_B`) and fuses their outputs using **Weighted Reciprocal Rank Fusion (RRF)**.

1.  **Run Config A (Content-Based):**
    * Typically uses rich document fields (`Title`, `OCR`, `Summary`).
    * Applies expansion techniques (e.g., `Same Box`) to infer folder relevance from document hits.
    * *Strengths:* Finds specific information buried deep in files.
2.  **Run Config B (Metadata-Based):**
    * Uses **`all_folders_folder_label=True`**. This ignores file content and retrieves based purely on the folder's semantic label (e.g., "Cuban Missile Crisis").
    * *Strengths:* Very high precision for broad topics; acts as a "sanity check" or baseline.
3.  **Fusion (RRF):**
    * The results from A and B are merged.
    * You can assign weights (e.g., `1.0` for Content, `0.65` for Metadata) to prioritize one strategy over the other.

**2. Usage Guide**

To create your own hybrid experiment, open `src/rrf_best_models.py` and modify the `run_hybrid_experiment` function.

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
python src/rrf_best_models.py
```

The results will be saved and evaluated automatically, ready for inspection in the Visualizer.

## SUSHI Visualizer Web Application

The **Visualizer** is the primary interface for analyzing experiment results and exploring the dataset. It is divided into two distinct applications, accessible via the sidebar navigation. It is a web application developed using Streamlit (Python library)

The tool is designed to bridge the gap between raw metric files and actionable insights, offering features like confidence interval visualization, topic-by-topic breakdowns, and dynamic filtering.

The SUSHI visualizer has two main screens:

- **Experiment Analyzer**: performance benchmarking;
- **Topics and Data Visualizer**: dataset exploration.

### Experiment Analyzer

This dashboard is a "Command Center" for evaluating models performances. It allows to answer two key questions: *"Which model is better?"* and *"Why is it better?"*.

**A. Model Global Performance (The Score Cards)**

At the top of the page, it's possible to see score cards for each model in the selected configuration (e.g., BM25 vs. ColBERT).

* **Mean nDCG@5:** The large number represents the average retrieval quality across all 45 topics. Higher is better (0.0 to 1.0).
* **Margin of Error (±):** The smaller number below shows the 95% Confidence Interval. If the intervals of two models overlap significantly, their performance difference might not be statistically significant.
* **N=30:** Indicates that the score is robust, calculated from 30 separate random trials (simulating different digitization scenarios).

**B. Topic Performance (The Dumbbell Chart)**

This chart  breaks down performance by individual topic. It is crucial for diagnosing "hard" vs. "easy" topics.

* **The Dot:** Represents the mean nDCG score for a specific topic.
* **The Line (Whiskers):** Represents the confidence interval. A long line means the model's performance was unstable across different random seeds (it depended heavily on *which* documents were digitized).
* **Comparison:** You will see multiple colored dots on the same line.
    * *Example:* If the **Orange Dot (ColBERT)** is consistently to the right of the **Blue Dot (BM25)**, the neural model is outperforming the keyword model on that specific topic.

**C. Cross-Experimental Table**

Located at the bottom, this table allows you to compare **different configurations** side-by-side (e.g., comparing a run with `Expansion` vs. a run `Without Expansion`).

* **Global nDCG:** The overall effectiveness of the run.
* **Global Rel:** The average number of relevant folders found in the top 5 results.
* **Per-Topic Columns:** Shows the detailed score for every single topic. Useful for spotting regressions (e.g., *"Did adding expansion hurt Topic 12?"*).

### Topics and Data Visualizer

This section is an explorer for the dataset itself (The "Ground Truth"). It helps you understand what the users are actually looking for and what the relevant documents look like.

**A. Topic Selection**

Select a Topic ID (e.g., `T1`) from the sidebar. You will see:

* **Title:** The short query (e.g., "Cuban Missile Crisis").
* **Description:** A sentence explaining the user's intent.
* **Narrative:** A detailed paragraph defining strictly what counts as relevant or irrelevant.

**B. Document View**

This tab shows all documents marked as relevant for the selected topic.

* **PDF Preview:** On the left, you can read the actual scanned document content.
* **Metadata Inspector:** On the right, you can see the file's indexed metadata (OCR text, Date, Box ID).
* **Sushi Folder Metadata:** Crucially, it shows the metadata of the *folder* this document belongs to, helping you understand the context of the match.

**C. Folder View**

This tab lists the "Gold Standard" folders — the physical folders that the user *should* have found.
* **Star Rating (⭐⭐):** Indicates the relevance level (High vs. Normal).
* **Enriched Labels:** You can see the full semantic label (e.g., `POLITICAL -> GENERAL -> ELECTIONS`) generated by the metadata enrichment process, which explains *why* this folder is relevant to the topic.

### Setup Experiments for the Visualizer

The Visualizer is built to be **dynamic**. It does not hardcode model names (except for colors); instead, it scans the file system to discover available experiments. To add a new model or experiment, you simply need to ensure your data follows the expected directory structure.

**1. Directory Structure**

All experiment results must live inside the `all_runs/` directory at the project root.

```text
ProjectRoot/
├── all_runs/
│   ├── TOFS_SB_TD_BM25/                 <-- Existing Run
│   └── TOFS_SB_TD_MY-NEW-MODEL/         <-- Your New Run (See naming convention below)
│       ├── model_overall_stats.json     <-- REQUIRED: Global aggregates (Mean/Margin)
│       ├── topics_mean_margin.json      <-- REQUIRED: Per-topic statistics
│       ├── topics_relevant_count_stats.json <-- REQUIRED: Relevance counts
│       ├── Random1_TopicsFolderMetrics.json   <-- REQUIRED: Individual seed data
│       ├── Random42_TopicsFolderMetrics.json
│       └── ...
```

**2. Folder Naming Convention**

The Visualizer parses folder names to automatically group experiments in the UI. You must follow this 4-part convention, separated by underscores: ```[SearchFields]_[ExpansionStrategy]_[QueryType]_[ModelName]```.

**Example: TOFS_SB_TD_MY-NEW-MODEL**

- **SearchFields (TOFS):** Title, Ocr, FolderLabel, Summary.
- **Expansion (SB):** Same Box.
- **Query (TD):** Title + Description.
- **Model (MY-NEW-MODEL):** The name that will appear in the charts.

⚠️ Important: If your model name has underscores (e.g., MY_NEW_MODEL), the parser will break. Use hyphens (-) for model names instead.

**3. Required JSON Files**

For the visualizer to render your charts, the following files must exist inside your run folder. These are automatically generated by the Evaluator class:

- *model_overall_stats.json:* Contains the single global "Mean nDCG" and "Margin of Error" used for the score cards.
- *topics_mean_margin.json:* Contains the [Min, Mean, Max] nDCG values for every topic. This drives the Dumbbell Chart.
- *Random{SEED}_TopicsFolderMetrics.json:* One file per random seed executed. Used to calculate the "N=" count and for detailed drill-downs.

**4. Registering a New Model Color**

While the system will automatically detect and list your new model, it won't know what color to assign it in the charts (it may default to a generic color or crash if strict mapping is on).

To assign a specific color to your new model:

- Open web_app/utils_experiments_viz.py.
- Locate the COLOR_MAP dictionary constant.
- Add your model name (exactly as it appears in the folder name) and a hex color code.

### How to Run

In order to run the SUSHI Experiment Runner, the follow steps must be followed:

1. **Install Python**: [https://www.python.org](https://www.python.org).
2. **Install Python libraries**: run the command ```pip install -r requirements.txt```. It is recommended to use a [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html) or a [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) env.
3. **Download all necessary files**: make sure to follow all the steps in the **Repository Setup** section of this README file.
4. **Setup experiments**: make sure the experiments are in the expected way.
4. **Run the Streamlit application**: now run the application and access it in the browser: ```streamlit run web_app/app_sushi.py```.

## Acknowledgements
This project was supported by the **[Fundação de Apoio à Pesquisa do DF](https://www.fap.df.gov.br/)**. We are grateful for their support.