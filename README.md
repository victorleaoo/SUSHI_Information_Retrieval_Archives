# SUSHI (Searching Unseen Sources for Historical Information) Test Collection and Experiments

## Index
- [About the Collection](#about-the-collection)
    - [Data Hierarchy: Boxes, Folders, and Documents](#data-hierarchy-boxes-folders-and-documents)
    - [Folder Metadata](#folder-metadata)
    - [Documents Metadata](#documents-metadata)
    - [Topics](#topics)
    - [Relevance judgment (QRels)](#relevance-judgment-qrels)
    - [Experiment Control Files (ECFs)](#experiment-control-files-ecfs)
- [Repository Setup](#repository-setup)
    - [Adding missing files](#adding-missing-files)
- [SUSHI Experiment Running](#sushi-experiment-running)
    - [1. System/File Architecture](#1-systemfile-architecture)
    - [2. The Experiment Workflow (Step-by-Step)](#2-the-experiment-workflow-step-by-step)
    - [3. Output Structure](#3-output-structure)
    - [4. Usage Example](#4-usage-example)
    - [5. Hybrid Models (Combining two different techniques with RRF) - `hybrid_models.py`](#5-hybrid-models-combining-two-different-techniques-with-rrf---hybrid_modelspy)
- [SUSHI Visualizer Web Application](#sushi-visualizer-web-application)
    - [Experiment Analyzer](#experiment-analyzer)
    - [Topics and Data Visualizer](#topics-and-data-visualizer)
    - [Setup Experiments for the Visualizer](#setup-experiments-for-the-visualizer)
    - [How to Run](#how-to-run)
- [Acknowledgements](#acknowledgements)

---

This repository presents the SUSHI Test Collection and provides a walk-through on how to access and use it. 

It also houses the code required to reproduce initial experiments, as well as a Python Streamlit web application for data and experiment visualization.

## About the Collection

The SUSHI Collection seeks to facilitate the development of archival Information Retrieval (IR). In many archival contexts, describing every individual document with metadata is impractical; often, the finest-grained descriptions available are for sets of boxes and folders, which may contain sparsely digitized documents. 

Consequently, the most common search task in an archive is not to find specific documents directly, but to identify *where* in the repository to look for them. Searchers typically identify promising boxes or folders, request them, and then physically or digitally browse the contents to find relevant information.

The first version of the NTCIR-18 SUSHI Collection is available at:  
[https://sites.google.com/view/ntcir-sushi-task/test-collection](https://sites.google.com/view/ntcir-sushi-task/test-collection)

### Data Hierarchy: Boxes, Folders, and Documents

The raw data structure follows a strict hierarchy representing the physical archival organization. The dataset contains **31,684 digitized documents**, stored in **1,336 folders** and **126 boxes**.

* **Download:** The raw box/folder/document structure is available [here](https://drive.google.com/file/d/1hA5FW0cNloi20coLlGvnv5wMap8ZN8YL/view?usp=sharing).
* **Format:** All documents are provided as PDF files.

**The Hierarchy:**

1. **Box:** Identified by a unique identifier (e.g., `N1234`).
2. **Folder:** Stored within a box, identified by a unique folder identifier (e.g., `N12345678`).
3. **Document:** Stored within a folder, identified by a unique SUSHI document identifier (e.g., `S12345.pdf`).

**Directory Structure:**

The test collection file system mirrors this hierarchy: one directory for each Box $\rightarrow$ one subdirectory for each Folder $\rightarrow$ one PDF file for each digitized Document. 

> **Note:** The PDF files contain embedded (uncorrected) OCR text, allowing the users of the collection to utilize raw text features in their experiments.

### Folder Metadata

Raw archival folder titles are often codes (e.g., `POL 15-1`). To make these searchable, we enrich them using the **Subject-Numeric Code (SNC)** systems from [1963](https://www.archives.gov/files/research/foreign-policy/state-dept/finding-aids/records-classification-handbook-1963.pdf) and [1965](https://www.archives.gov/files/research/foreign-policy/state-dept/finding-aids/dos-records-classification-handbook-1965-1973.pdf).

For NTCIR, it was provided the FoldersV1.2.json file, however additions were provided in order to have more fields (version FoldersV1.3.json): the labels from 1965 and 1963 (not only the newest one available); the raw version of scope note; the truncated version of scope note; the parent snc (if possible) and its recursive expansion.

This enrichment is handled by the `FolderLabelConstructor` class ([src/data_creation/SNCLabelTranslate.py](https://github.com/victorleaoo/SUSHI_Information_Retrieval_Archives/blob/main/src/data_creation/SNCLabelTranslate.py)), merging raw folder data with the SNC Translation Table.

The file can be found at [FoldersV1.3.json](https://github.com/victorleaoo/SUSHI_Information_Retrieval_Archives/blob/main/data/folders_metadata/FoldersV1.3.json).

* **Input Source:** [SncTranslationV1.3.xlsx](https://github.com/victorleaoo/SUSHI_Information_Retrieval_Archives/blob/main/data/folders_metadata/SncTranslationV1.3.xlsx)
    * `SNC`: The code (e.g., `POL 1`).
    * `1965`: Label from the 1965 classification (Preferred).
    * `1963`: Label from the 1963 classification (Fallback).
    * `Scope Note`: Description of the category's inclusion/exclusion criteria.

**Output Fields (JSON):**

| Field | Description | Purpose |
| :--- | :--- | :--- |
| `label1965` | Literal text from 1965 column. | Preferred term source. |
| `label1963` | Literal text from 1963 column. | Fallback source. |
| `main_title` | Resolved primary label (1965 or 1963). | Core display name for the folder. |
| `raw_scope` | Full scope note text. | Provides maximum context. |
| `scope_truncated` | Cleaned scope note. "Stopper Keywords" (e.g., `SEE`, `Exclude`) are used to cut off negative definitions. | High-precision context; removes confusing negative keywords. |
| `parent` | SNC code of the immediate parent. | hierarchy traversal. |
| `label_parent_expanded` | Full semantic path string. | e.g., Resolves `POL 15-1` to "POLITICAL -> GENERAL" rather than just "General". |

### Documents Metadata

The **Document Metadata** file aggregates information available for each document, sourcing data from NARA and Brown University records. It can be loaded at [itemsV1.2.json](https://drive.google.com/file/d/1c_hpR_lgdGeskXaNQTdCS7nO9s1R2NOb/view?usp=share_link).

* **Input Source:** [SubtaskACollectionMetadataV1.1.xlsx](https://github.com/victorleaoo/SUSHI_Information_Retrieval_Archives/blob/main/data/items_metadata/SubtaskACollectionMetadataV1.1.xlsx)
    * Contains SUSHI unique identifiers.
    * Merges available original NARA and Brown metadata.

**Output Fields (JSON):**

| Field | Description |
| :--- | :--- |
| `Sushi Box` | The SUSHI unique ID for the box. |
| `Sushi Folder` | The SUSHI unique ID for the folder. |
| `Sushi File` | The SUSHI unique ID for the document (including `.pdf` extension). |
| `NARA <metadata>` | The set of raw fields provided by NARA. |
| `Brown <metadata>` | The set of raw fields provided by Brown. |
| `date` | The resolved date of the file (prioritizes Brown metadata if available, falls back to NARA). |
| `title` | The resolved title of the file (prioritizes Brown metadata if available, falls back to NARA). |
| `ocr` | A list containing the OCR text for each page. Generated primarily using ABBYY FineReader. |
| `summary` | A generated summary of the document content produced by GPT4o. |

### Topics

The [Topics File](src/data_creation/topics_output.txt) contains 45 information needs. Each topic includes:

* **ID:** Unique identifier.
* **Title:** A short, web-search-style query.
* **Description:** A single sentence succinctly expressing the information need.
* **Narrative:** A paragraph providing detailed criteria for relevance assessment.

### Relevance judgment (QRels)

Relevance assessments are provided at three granularity levels in the `qrels/formal-run-qrels/` directory:

- [Box Qrels](https://github.com/victorleaoo/SUSHI_Information_Retrieval_Archives/blob/main/qrels/formal-box-qrel.txt);
- [Folder Qrels](https://github.com/victorleaoo/SUSHI_Information_Retrieval_Archives/blob/main/qrels/formal-folder-qrel.txt);
- [Document Qrels](https://github.com/victorleaoo/SUSHI_Information_Retrieval_Archives/blob/main/qrels/formal-document-qrel.txt).

**Format:**

Each line maps a topic to a Box/Folder/Document ID with a relevance score:

* **3:** Highly Relevant
* **1:** Somewhat Relevant
* **0:** Not Relevant

*(Unjudged items are omitted from the files).*

**Note on Documents Qrels Score Normalization:**

The [raw judgments for documents](./qrels/OLD-formal-document-qrel.txt) contained a wider range of labels. These were normalized to the standard 3-point scale as follows:

| Original Assessor Label | Final Qrels Score | Meaning |
| :--- | :--- | :--- |
| **4** | **3** | Highly Relevant |
| **3** | **1** | Relevant |
| **1, 2** | **0** | Not Relevant |
| **-1** | *(Removed)* | Not judgeable |

### Experiment Control Files (ECFs)

The **Experiment Control File (ECF)** serves the role of a standard "Topics" file in TREC evaluations but is adapted for the SUSHI task. Real-world archives have limited digitization. SUSHI simulates this by explicitly defining which documents are "visible" (digitized/training data) to the system for a given topic.

**Structure:**

Each ECF defines specific **Experiment Sets**. An experiment set maps a list of **Topics** to a specific list of **TrainingDocuments**.

* **TrainingDocuments:** A slash-delimited path (`SushiBox/SushiFolder/SushiFile`, e.g., `N1234/N12345678/S12345.pdf`). These are the only documents the system is allowed to "read" to learn the relevance features for the associated topics.
* **Topics:** The queries for which the system must find *other* relevant folders (containing documents *not* in the training set).

**ECF Conditions:**

We provide [ECFs](https://github.com/victorleaoo/SUSHI_Information_Retrieval_Archives/tree/main/ecf/random_generated) covering three different experimental conditions:

1.  **[All Training Documents (No Mask)](https://github.com/victorleaoo/SUSHI_Information_Retrieval_Archives/blob/main/ecf/random_generated/ECF_ALL_TRAINING_SET.json):** * ECF containing all available training documents for all topics.
2.  **Random (Uniform):** * ECF simulating a uniform digitization strategy (e.g., 5 documents per box), selected via different random seeds. These are the ECFs that have the name such as *ECF_RANDOM_seed.json*.
3.  **Uneven (Skewed):** * ECF simulating a skewed distribution of digitized documents per box. These are the ECFs that have the name such as *ECF_UNEVEN_Random_Seed_seed.json*.
    * The distribution logic is detailed in [src/RGdistribution.xlsx](https://github.com/victorleaoo/SUSHI_Information_Retrieval_Archives/blob/main/src/RGdistribution.xlsx).

*The ECF creation logic is located at [create_random_ecf](https://github.com/victorleaoo/SUSHI_Information_Retrieval_Archives/blob/main/src/data_loader.py#L77).*

---

## Repository Setup

To run the applications correctly, it is necessary to reproduce the data structure locally, as large or sensitive files are not versioned in this repository (they are git-ignored).

```
├── all_runs/ # It contains runs that were already made from models  
├── data/                           
│   ├── folders_metadata/
│   │   └── FoldersV1.3.json
│   ├── items_metadata/
│   │   └── itemsV1.2.json  # ⚠️ Download Document metadata
│   ├── raw/                # ⚠️ Download and Place the raw Box/Folder structure with PDFs here
│   │   └── A0001
│   │   └── A0002
│   │   └── A0003
│   │   └── ...
├── ecf/                            
├── qrels/                          
│   ├── formal-box-qrel.txt
│   ├── formal-document-qrel.txt
│   └── formal-folder-qrel.txt
├── src/     # Experiment Run Generator
├── web_app/ # SUSHI visualizer
├── .gitignore
├── README.md
└── requirements.txt
```

### Adding missing files

All files can be found at the [SUSHI Test Collection](https://sites.google.com/view/ntcir-sushi-task/test-collection):

- [itemsV1.2.json](https://drive.google.com/file/d/1c_hpR_lgdGeskXaNQTdCS7nO9s1R2NOb/view?usp=share_link)
- [raw](https://drive.google.com/file/d/1hA5FW0cNloi20coLlGvnv5wMap8ZN8YL/view?usp=sharing)

After downloading, reproduce the following steps:

1. Create a dir called ```data```;
2. Inside the dir ```data/items_metadata```, place the ```itemsV1.2.json``` file;
3. Inside the dir ```data/raw```, place the ```sushi-files.zip``` inside it and unzip it. After unziping, bring all folders out to the ```raw``` dir and delete the ```sushi-files``` folder and the ```sushi-files.zip``` file.

---

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

## [SUSHI Visualizer Web Application](https://tinyurl.com/sushisigir)

The **SUSHI Visualizer** is the main interactive interface for analyzing experiment runs and exploring the archival collection. Built with Streamlit, it combines experiment benchmarking, topic inspection, metadata exploration, and ECF analysis in a single application. The app is launched from the [web_app/app_sushi.py](web_app/app_sushi.py) entry point and can be accessed at [https://tinyurl.com/sushisigir](https://tinyurl.com/sushisigir).

The interface is organized around a left-side navigation menu, where each entry exposes a different analysis workflow.

### Side Menu Navigation

The side menu contains five main sections:

1. **How-To Guide**
   - Introduces the SUSHI task, the box/folder/document hierarchy, the concept of sparse digitization, and the meaning of evaluation metrics such as nDCG@5 and relevance grades.

2. **Experiment Analyzer**
   - The main dashboard for comparing retrieval runs.
   - Supports three complementary analysis modes:
     - **Single Experiment Analysis**: select a configuration and compare the models contained in that experiment.
     - **Retrieval Analysis**: inspect the top-ranked folders for each topic and review qrels grades for those folders.
     - **Two-Experiment Comparison**: compare any two runs side-by-side, including a Wilcoxon signed-rank test and a topic-level overlay chart.

3. **Data Overview**
   - A collection exploration page for understanding the dataset beyond the retrieval metrics.
   - Includes:
     - collection statistics such as number of folders, documents, boxes, and SNCs;
     - histograms for documents per folder and folders per box;
     - SNC distribution views at three granularity levels (3-level, 2-level, and primary SNC);
     - a deep-dive view to inspect folder and document content by selected SNC;
     - document analysis tools to browse documents by SNC or by folder.

4. **Topic Viewer**
   - Lets the user browse the 45 evaluation topics and inspect their descriptions, narratives, and relevant gold-standard items.
   - The page builds a hierarchical view of the relevant structure: box → folder → document.
   - Each relevant item is shown with star-based relevance grades, and the viewer can reveal metadata such as SNC, scope notes, summaries, and OCR text.

5. **ECF Inspector**
   - Focuses on the Experiment Control Files (ECFs), which define which documents are visible during training.
   - Provides multiple views for understanding coverage:
     - overall coverage statistics;
     - coverage by SNC;
     - coverage by folder;
     - coverage by box;
     - relevance coverage for the 45 topics;
     - direct comparison between two ECFs.

### Experiment Analyzer in Detail

This is the main page for answering the core research questions: “Which model performs better?” and “Why?”. It is designed for comparative analysis rather than simple browsing.

#### Inputs and run selection

The app discovers available experiment results from the [all_runs](all_runs) directory and groups them by configuration. Before looking at the charts, users should first select:

- a subfolder/scope to narrow the candidate runs;
- a target experiment configuration;
- one or two specific runs for comparison.

This selection step matters because the best interpretation comes from comparing runs that share a similar setup and differ mainly in the retrieval strategy or expansion method.

#### What is shown in the single-experiment view

- **Global performance cards**: show the mean nDCG@5, the margin of error, and the number of seeds contributing to the estimate. These are the first indicators of whether a method is consistently strong.
- **Topic-level dumbbell chart**: displays each topic’s mean score and its confidence interval. This is useful for understanding whether a model is uniformly good or only strong on a few topics.
- **Seed variance view**: shows how much the score changes across random seeds. A narrow distribution suggests stable behavior; a wide distribution suggests that the outcome depends heavily on the sampled training documents.

<p align="center">
  <img src="img/single_experiments.png" alt="Single Experiments" width="900" />
</p>

<p align="center">
  <img src="img/comparison_experiments.png" alt="Comparison Experiments" width="900" />
</p>

#### Run naming convention

Experiment folders are interpreted using a four-part naming convention:

- **Search fields**
- **Expansion strategy**
- **Query type**
- **Model name**

For example, a folder such as `TOFS_SB_TD_BM25` indicates:

- `TOFS`: title/ocr/folderlabel/summary-based search fields;
- `SB`: same-box expansion;
- `TD`: title + description query fields;
- `BM25`: the retrieval model name.

### Data Overview in Detail

This section is intended for dataset exploration rather than model benchmarking. It helps explain the collection conditions behind the retrieval metrics and is particularly useful when you suspect that the data distribution is affecting the results.

#### What is shown in the collection statistics

The opening view displays:

- total folders, documents, and boxes;
- how many distinct SNCs exist at different levels;
- how many folders contain scope notes;
- distributional summaries such as OCR page counts and the number of documents per folder.

These metrics help you understand whether the collection is balanced or skewed, which is important because retrieval performance can be strongly influenced by the underlying metadata structure.

<p align="center">
  <img src="img/data_overview1.png" alt="Data Overview Stats" width="900" />
</p>

#### What is shown in the SNC exploration views

The SNC tabs provide a structured view of the archival classification system:

- **3-Level SNC**: the most detailed view, useful for fine-grained analysis.
- **2-Level SNC**: a middle layer that can reveal broader semantic clusters.
- **1-Level (Primary)**: a broader overview of the main classification families.

Each tab includes tables, histograms, and bar charts. The deep-dive view also allows you to inspect the folders and documents attached to a selected SNC.

<p align="center">
  <img src="img/data_overview2.png" alt="Data Overview SNC" width="900" />
</p>


#### What is shown in the document browsing tabs

The document analysis tabs let you:

- inspect keyword distributions in titles and summaries;
- browse documents grouped by SNC;
- browse all documents belonging to a selected folder.

### Topic Viewer in Detail

The Topic Viewer is designed for qualitative inspection of the relevance judgments and the archival context behind each topic. It is the best place to move from metric-driven analysis to understanding why a certain result is considered correct or incorrect.

#### What is shown on this page

For a selected topic, the page displays:

- the topic title, description, and narrative;
- a summary of how many relevant boxes, folders, and documents are associated with the topic;
- a hierarchical expansion of the relevant structure:
  - box-level grouping;
  - folder-level metadata and scope notes;
  - document-level summaries and OCR previews.

This gives you the actual archival context behind the evaluation topic, not just the score.

<p align="center">
  <img src="img/topic_viewer.png" alt="Topic Viewer" width="900" />
</p>

### ECF Inspector in Detail

The ECF Inspector focuses on the experimental conditions used to train the system. It helps answer questions such as: Which documents were visible to the model during training, and how much of the collection was actually available to learn from?

#### What is shown in the overview and coverage views

The app presents several complementary views:

- **Overview**: a histogram of documents per covered folder.
- **By SNC**: a table of coverage by classification code, including counts of SNCs with and without documents.
- **By Folder**: a detailed inspection of the folders included in the selected ECF.
- **By Box**: box-level coverage statistics.
- **Relevance Coverage**: how much of the relevant-folder set for the 45 topics is covered by the ECF.
- **Compare ECFs**: a side-by-side view to compare the training visibility of two different experimental settings.

<p align="center">
  <img src="img/ecf_inspector.png" alt="ECF Inspector" width="900" />
</p>

<p align="center">
  <img src="img/ecf_comparison.png" alt="ECF Comparison" width="900" />
</p>

### Setup for New Experiments and Visualizer Inputs

The visualizer is dynamic and discovers available experiment results from the filesystem. To add a new run, make sure that the output files follow the expected structure under the [all_runs](all_runs) directory.

#### Required directory layout

```text
ProjectRoot/
├── all_runs/
│   ├── TOFS_SB_TD_BM25/
│   └── TOFS_SB_TD_MY-NEW-MODEL/
│       ├── model_overall_stats.json
│       ├── topics_mean_margin.json
│       ├── topics_relevant_count_stats.json
│       ├── Random1_TopicsFolderMetrics.json
│       └── ...
```

#### Required files

- `model_overall_stats.json`: global mean nDCG and margin of error.
- `topics_mean_margin.json`: per-topic statistics used by the dumbbell chart.
- `topics_relevant_count_stats.json`: relevance counts used by the comparison views.
- `Random{SEED}_TopicsFolderMetrics.json`: one file per seed for detailed analysis and N-count computation.

#### Naming convention

Use a four-part folder name separated by underscores:

```text
[SearchFields]_[ExpansionStrategy]_[QueryType]_[ModelName]
```

Example:

```text
TOFS_SB_TD_MY-NEW-MODEL
```

If the model name contains underscores, replace them with hyphens to avoid parsing issues.

#### Optional color registration

If you add a new model name and want it to appear with a specific color in the charts, add it to the `COLOR_MAP` dictionary in [web_app/utils_experiments_viz.py](web_app/utils_experiments_viz.py).

### How to Run

To run the application locally:

1. Install Python from [https://www.python.org](https://www.python.org).
2. Install the dependencies with:

   ```bash
   pip install -r requirements.txt
   ```

3. Follow the repository setup steps described earlier in this README so the required data files are available.
4. Start the Streamlit app from the [web_app](web_app) directory:

   ```bash
   streamlit run app_sushi.py
   ```

---

## Acknowledgements

Development of the SUSHI test collection was supported in part by Japan Society for the Promotion of Science KAKENHI Grant 23KK0005 and National Institute of Informatics Open Collaborative Research 2024 (24S0505).

In addition, part of the work was also supported by Fundação de Apoio e Pesquisa do Distrito Federal (FAPDF).