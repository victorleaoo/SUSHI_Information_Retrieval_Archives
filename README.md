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
    - [2. Configuration Arguments](#2-configuration-arguments)
    - [3. The Experiment Workflow (Step-by-Step)](#3-the-experiment-workflow-step-by-step)
    - [4. Output Structure](#4-output-structure)
    - [5. Running a Single Custom Experiment](#5-running-a-single-custom-experiment)
    - [6. Running the Full Experiment Matrix - `run_new_experiments.py`](#6-running-the-full-experiment-matrix---run_new_experimentspy)
    - [7. Hybrid Models (Combining two different techniques with RRF) - `hybrid_models.py`](#7-hybrid-models-combining-two-different-techniques-with-rrf---hybrid_modelspy)
    - [8. BM25F Field-Weight Tuning - `tuning_bm25/bm25_tuning.py`](#8-bm25f-field-weight-tuning---tuning_bm25bm25_tuningpy)
    - [9. Wilcoxon Significance Testing](#9-wilcoxon-significance-testing)
    - [10. Four-Way ANOVA](#10-four-way-anova)
- [SUSHI Visualizer Web Application ("SUSHI BAR")](#sushi-visualizer-web-application-sushi-bar)
    - [Application Pages](#application-pages)
    - [Adding a New Experiment to the Visualizer](#adding-a-new-experiment-to-the-visualizer)
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

This system is designed to simulate Information Retrieval scenarios on sparsely digitized archival collections (SUSHI). It runs experiments by sampling subsets of the collection (using random seeds), training models on those subsets, expanding/fusing results for folders that have no documents in the ECF, and evaluating the performance.

All commands in this section assume the working directory is `src/` (`cd src`), since `DataLoader` loads `RGdistribution.xlsx` via a path relative to the current working directory, and `RunGenerator` writes its output to `../all_runs/`.

### 1. System/File Architecture

The pipeline workflow is built on four modular classes, each with a distinct responsibility.

- **1. RunGenerator (*src/run_generator.py*)**
    * **Role:** The main controller. It manages the experiment loop, initializes models, coordinates data flow, and executes the retrieval pipeline.
    * **Responsibility:** It delegates tasks to the helper classes below, runs RRF fusion between models, applies the folder-expansion heuristics, and drives one of three protocols: `'random'` (30-seed sparse sampling), `'all_documents'` (Oracle, single pass), or `'official_ecf'` (the NTCIR-18 protocol, 3 fixed ExperimentSets).

- **2. DataLoader (*src/data_loader.py*)**
    * **Role:** Manages file I/O and data structure.
    * **Responsibility:**
        * Loads metadata (`FoldersV1.3.json`, `itemsV1.2.json`) and the topics file (`src/data_creation/topics_output.txt`).
        * Maps the physical directory structure (`Box -> Folder -> File`) by scanning `data/raw/`.
        * Generates the **ECF (Experimental Collection Format)** on the fly via `create_random_ecf(seed, sampling, docs_per_box)`: samples training documents per box for a given random seed, either `'uniform'` (a fixed number of docs per box, round-robin across folders) or `'uneven'` (skewed per-box targets read from `src/RGdistribution.xlsx`).
        * Loads two pre-built ECF files instead of sampling: `load_all_docs_ecf()` reads `ecf/random_generated/ECF_ALL_TRAINING_SET.json` (every document — used by `run_type='all_documents'`), and `load_official_ecf()` reads `ecf/ntcir18/Ntcir18SushiOfficialExperimentControlFileV1.1.json` (the official 3-ExperimentSet protocol).

- **3. RetrievalModel (*src/models.py*)**
    * **Role:** Abstract base class for search algorithms.
    * **Subclasses:**
        * `BM25Model`: Uses **PyTerrier** for sparse, frequency-based retrieval. Automatically switches from BM25 to **BM25F** (field-weighted) when more than one searching field is given. The `tuned_weights` flag picks between hand-tuned field weights (`BM_25_FIELD_WEIGHTS` in `models.py`, run notation `'L'` for Learned) and PyTerrier's defaults (run notation `'B'` for unweighted/Baseline).
        * `EmbeddingsModel`: Uses **SentenceTransformers** (`all-mpnet-base-v2` by default) for dense vector retrieval via Cosine Similarity. Automatically picks CUDA, then Apple MPS, then CPU.
        * `ColBERTModel`: Uses **PyLate** (`lightonai/colbertv2.0`) for late-interaction retrieval (token-to-token MaxSim) with a **PLAID** index built under `src/pylate-index/`.
    * **Standard:** Every model must implement `train(training_data)` to build an index and `search(query)` to return a DataFrame with `docno`, `folder`, `score`.

- **4. Evaluator (*src/evaluator.py*)**
    * **Role:** Computes performance metrics.
    * **Responsibility:**
        * Converts model outputs into a standard **TREC run file** (`results/RunResults.tsv`, overwritten on every seed/run).
        * Compares results against the folder QRELs (`qrels/formal-folder-qrel.txt`) using `pytrec_eval` (`ndcg_cut`, `map`, `recip_rank`, `success`).
        * Calculates a custom metric: `count_relevant_top5`, how many relevant folders appear in the top 5.
        * `generate_aggregated_metrics()` aggregates results across seeds (for `'random'` runs) or reads the single output file (for `'all_documents'` / `'official_ecf'` / ALLFL runs) to produce per-topic Mean/95% CI and a global Mean nDCG@5.

### 2. Configuration Arguments

`RunGenerator` (in `src/run_generator.py`) is highly configurable. Here is what each constructor argument controls:

| Argument | Type | Description |
| :--- | :--- | :--- |
| `searching_fields` | `List[List[str]]` | Which metadata fields to index, one list per configuration to run. Ex: `[['title', 'ocr']]`. If multiple fields are provided, BM25 upgrades to BM25F automatically. Valid fields: `'title'`, `'ocr'`, `'folderlabel'`, `'summary'`. |
| `query_fields` | `List[str]` | Which parts of the Topic to use as the query.<br>`'T'`: Title only.<br>`'TD'`: Title + Description.<br>`'TDN'`: Title + Description + Narrative. |
| `run_type` | `str` | `'random'` (default): Loops over the 30 fixed seeds in `RANDOM_SEED_LIST`, sampling a new training set each time.<br>`'all_documents'`: Single pass using the entire collection (Oracle mode), via the pre-built `ECF_ALL_TRAINING_SET.json`.<br>`'official_ecf'`: Single pass following the NTCIR-18 official protocol — trains and evaluates independently on each of the 3 official ExperimentSets (15 topics each), then merges all 45 topic results into one run. |
| `models` | `List[str]` | The models to ensemble. Options: `'bm25'`, `'embeddings'`, `'colbert'`. If more than one is given, results are fused with Reciprocal Rank Fusion (RRF). |
| `sampling` | `str` | Only used when `run_type='random'`. `'uniform'` (default): `docs_per_box` documents sampled from every box. `'uneven'`: skewed per-box document counts read from `src/RGdistribution.xlsx`. |
| `docs_per_box` | `int` | Target documents sampled per box under uniform sampling (default `5`). |
| `expansion` | `List[str]` | Strategies used to infer scores for folders with no directly retrieved documents (see `create_folder_relations_for_expansion` / `produce_expansion_results` in `run_generator.py`).<br>`'same_box'`: Neighbor is in the same box.<br>`'same_snc'`: Neighbor has the exact same Classification Code (SNC).<br>`'close_date'`: Neighbor has the same SNC and its date falls within the folder's date range.<br>`'similar_snc'`: Neighbor's SNC shares the same top-level (and, for `POL`, second-level) code.<br>`[]`: No expansion. When more than one technique is given, a folder is only scored if it has neighbors satisfying **all** of them (set intersection). |
| `all_folders_folder_label` | `bool` | If `True`, ignores document contents entirely and retrieves based only on folder metadata labels (`label_parent_expanded` + `scope_truncated`) for every folder in the collection — one training "document" per folder, independent of any ECF sampling. `searching_fields` should be `[['folderlabel']]` in this mode. This is the "ALLFL" (All-Folder-Label) mode used as a metadata-only baseline and as one side of the hybrid fusion in `hybrid_models.py`. |
| `rrf_input` | `str` | Only relevant when `len(models) > 1`. `'docs'` (default): fuses model results at the *document* level first, then expands the fused list to folders. `'folders'`: expands each model's results to folders independently, then fuses the folder-level scores. |
| `expansion_ceiling_k` | `int` | **Safety ceiling.** The rank `k` (1-indexed) that inferred/expanded folder scores cannot surpass among the directly-retrieved results; if an inferred score would land above that rank, all inferred scores are penalized down just below it. Default `2`. |
| `bm25_tuned` | `bool` | Passed through to `BM25Model`. `True` (default) = Learned/tuned BM25F field weights (`'L'`); `False` = PyTerrier default/unweighted BM25F (`'B'`). Only affects multi-field runs. |

### 3. The Experiment Workflow (Step-by-Step)

By calling `RunGenerator.run_experiments()` (or `run_single_seed()` / `run_official_ecf()` directly), the system follows this lifecycle:

**Phase A: Setup**

1.  **Configuration:** The system reads the parameters (fields to index, models to use, expansion techniques, run type).
2.  **Directory Prep:** Creates a unique output folder under `../all_runs/` based on the config, via `saving_folder_name()` (see [Output Structure](#4-output-structure) below).

**Phase B: The Simulation Loop**

- For `run_type='random'` (the default), the system iterates through the 30 fixed random seeds in `RANDOM_SEED_LIST`. For each seed:
    1.  **Sampling (ECF Creation):** `DataLoader.create_random_ecf()` samples `docs_per_box` documents from every box (uniform) or a skewed per-box count (uneven). These documents become the "Training Set"; everything else is "undigitized" and invisible to model training.
    2.  **Model Training:** The active models (`BM25`, `ColBERT`, `Embeddings`) each build an index using only the sampled training set (or, in ALLFL mode, one "document" per folder built from folder metadata).
    3.  **Retrieval:** For each of the 45 topics, a query is built from `TITLE`/`DESCRIPTION`/`NARRATIVE` according to `query_fields`, and every active model searches its index.
    4.  **Fusion & Expansion:**
        * **RRF:** If more than one model is active, results are merged with weighted Reciprocal Rank Fusion (weights in `RFF_WEIGHTS`), either at the document level or the folder level depending on `rrf_input`.
        * **Expansion:** "Ghost folders" not directly retrieved are scored by averaging the scores of related training documents (same box / same SNC / close date / similar SNC, per `expansion`), then capped by `expansion_ceiling_k`.
    5.  **Evaluation:** The ranked folder list per topic is saved to a TREC run file and evaluated against the QRELs.
    - For `all_folders_folder_label=True`, since the ALLFL training set is entirely seed-independent, only a single pass is run (seed `0`) instead of looping over all 30 seeds.
- For `run_type='all_documents'`, a single pass is run using every document in the collection (via `ECF_ALL_TRAINING_SET.json`) — no sampling, no seed loop.
- For `run_type='official_ecf'` (`run_official_ecf()`), the system loops over the 3 official `ExperimentSets` (15 topics each), training the active models independently on each set's `TrainingDocuments`, then concatenates all 45 topic results into one run file before evaluating.

After the run(s) complete, `evaluator.generate_aggregated_metrics()` writes the aggregated per-topic and global statistics described below.

### 4. Output Structure

Every run writes its output under `../all_runs/<run_folder_name>/`, where `run_folder_name` comes from `RunGenerator.saving_folder_name()` by default (e.g. `4perBox-TOFS_SB-2_TD_BM25-COLBERT`, or `OfficialECF-4perBox-...` for the official protocol) — **unless** the caller passes an explicit custom folder name, which is what `run_new_experiments.py` does for every experiment in this repository's `all_runs/` directory (see [section 6](#6-running-the-full-experiment-matrix---run_new_experimentspy) for that naming convention). Inside the run folder:

1.  **`Random{SEED}_TopicsFolderMetrics.json`** *(`run_type='random'` only, one per seed)*:
    * Detailed `pytrec_eval` metrics (`ndcg_cut_5`, `ndcg_cut_10`, `map`, `recip_rank`, `success`, ...) plus `count_relevant_top5` for every topic.
2.  **`AllDocuments_TopicsFolderMetrics.json`** / **`OfficialECF_TopicsFolderMetrics.json`** / **`AllFolderLabel_TopicsFolderMetrics.json`**:
    * The single-file equivalent of the above for `'all_documents'`, `'official_ecf'`, and ALLFL runs respectively.
3.  **`topics_values.json`**:
    * Raw list of `ndcg_cut_5` values per topic across every seed (or a single-item list for non-`'random'` runs).
4.  **`topics_mean_margin.json`**:
    * The `[lower, mean, upper]` 95% confidence interval for *each topic*, derived from `topics_values.json`. This drives the dumbbell charts in the Visualizer.
5.  **`topics_relevant_count_stats.json`**:
    * The mean `count_relevant_top5` for each topic.
6.  **`model_overall_stats.json`** (`'random'` runs) or **`all_documents_model_overall_stats.json`** (`'all_documents'` / `'official_ecf'` / ALLFL runs):
    * The **Global Mean nDCG@5** and its **95% Margin of Error**, averaged across all topics (and, for `'random'`, across all seeds).

### 5. Running a Single Custom Experiment

To run one experiment configuration — e.g. **BM25 and ColBERT, searching Title and OCR, using 'Same Box' expansion, fusing at the document level**:

1. **Install Python**: [https://www.python.org](https://www.python.org).
2. **Install Python libraries**: run the command ```pip install -r requirements.txt```. It is recommended to use a [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html) or a [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) env.
3. **Install Java for Pyterrier**: [https://pyterrier.readthedocs.io/en/latest/troubleshooting/java.html](https://pyterrier.readthedocs.io/en/latest/troubleshooting/java.html)
    - If you're running on Windows, set `JAVA_HOME` inside `_init_pyterrier()` in `src/models.py`.
4. **Run it from the `src/` directory** — either edit and run `python3 run_generator.py` directly (its `__main__` block instantiates a default `RunGenerator` and calls `run_experiments()`), or create a new script alongside it:

```python
# from within src/
from run_generator import RunGenerator

# Configure the experiment
gen = RunGenerator(
    searching_fields=[['title', 'ocr']],
    query_fields=['TD'],
    run_type='random',
    models=['bm25', 'colbert'],
    expansion=['same_box'],
    rrf_input='docs',
    expansion_ceiling_k=2
)

# Execute
gen.run_experiments()
```

**IMPORTANT NOTE**: the code doesn't automatically delete the `terrierindex/` folder created for each BM25 run, therefore, it is necessary to **manually delete it after the run generator stops** running.

### 6. Running the Full Experiment Matrix - `run_new_experiments.py`

`src/run_new_experiments.py` is the driver script actually used to produce every run currently in `all_runs/` on this branch. It wraps `RunGenerator` to re-run a fixed matrix of configurations across all 3 query fields, a docs-per-box sweep, and the official-ECF protocol — with resumability, so an interrupted batch can simply be re-launched.

Run it from the `src/` directory:

```bash
cd src
python run_new_experiments.py              # Run ALL experiments (core + sweep + official)
python run_new_experiments.py 1 3 5        # Run specific experiment ids
python run_new_experiments.py T TDN        # Run every core config for those query fields
python run_new_experiments.py DOCSBOX      # Run the full U1-U4/K5/A- docs-per-box sweep (all query fields)
python run_new_experiments.py OFFICIAL     # Run the offset 1-8 official-ECF re-runs (all query fields)

# example used to run the full matrix unattended:
nohup python run_new_experiments.py > ../new_experiments.log 2>&1 &
```

**What it runs:**

- **15 core configurations x 3 query fields (T / TD / TDN) = 45 experiments**, each under the `'random'` protocol (30 seeds). The 15 configurations are defined in the `CONFIGS` table at the top of the file and span: tuned/untuned BM25F alone (`L`/`B`), BM25F on a single field at a time (OCR-only, Summary-only, Title-only, FolderLabel-only), ColBERT-only (`C`), Embeddings-only (`E`), untuned/tuned 3-model RRF (`Z`/`W`), the ALLFL ColBERT baseline, and several **hybrid** configurations that RRF-fuse a document-level ranker with an ALLFL (folder-label-only) ranker (`V`, `X`, `W.-.c`, `W.s---2.c`) using `hybrid_models.perform_hybrid_fusion`.
- **A docs-per-box sweep (6 variants x 3 query fields = 18 experiments)**, all using the tuned 3-model RRF (`W`) configuration on TOFS fields, varying only how the training ECF is sampled: `U1`-`U4` (uniform, 1-4 docs/box), `K5` (uneven sampling), and `A-` (the `'all_documents'` oracle run, single pass).
- **Official-ECF re-runs of the offset-1..8 configurations (8 x 3 query fields = 24 experiments)**, using `run_type='official_ecf'` instead of the 30-seed random protocol (folders get an `.OfficialECF` suffix).

Every produced run folder follows a **6-part, dot-separated name**: `<Sample>.<QueryField>.<Scoring>.<Fields>.<Expansion>.<LabelFusion>` — e.g. `U5.TD-.L.TOFS.mx--2.-` or `U5.T--.W.TOFS.-.c`. Roughly:
- **Sample**: `U5`/`U1`-`U4` (uniform sampling, N docs/box), `K5` (uneven sampling), `A-` (all-documents oracle).
- **QueryField**: `T--`, `TD-`, or `TDN`.
- **Scoring**: the active ranker(s) — `B` (untuned BM25F), `L` (tuned BM25F), `C` (ColBERT), `E` (Embeddings), `W` (tuned BM25F+Embeddings+ColBERT RRF), `Z` (untuned version), `V`/`X` (BM25F+ColBERT hybrids), or `-` for the ALLFL-only baseline.
- **Fields**: a 4-character `TOFS`-style mask (Title/OCR/FolderLabel/Summary), `----` when only folder labels are used.
- **Expansion**: encodes which expansion technique(s) and `expansion_ceiling_k` were used, or `-` for none.
- **LabelFusion**: `-` if no ALLFL label ranker was fused in, `b`/`c` if the run was RRF-fused with an ALLFL BM25/ColBERT ranker.

See the `CONFIGS` list and the module docstring in `run_new_experiments.py` for the exact mapping of every id/suffix combination — it is the authoritative reference for what each run folder in `all_runs/` represents. A run is considered already done (and skipped on re-invocation) if its output folder contains `model_overall_stats.json` or `all_documents_model_overall_stats.json`.

### 7. Hybrid Models (Combining two different techniques with RRF) - `hybrid_models.py`

`src/hybrid_models.py` provides `perform_hybrid_fusion()`, used both standalone and by `run_new_experiments.py`'s hybrid configurations, to **fuse distinct retrieval strategies** into a single ranking. While a single `RunGenerator` ensembles models that share the same configuration (e.g., BM25 + ColBERT both indexing the same document text), this lets you combine fundamentally different approaches — most commonly, **Document Retrieval** (OCR/title/summary content) with **Folder Retrieval** (ALLFL, folder-label-only).

**1. How It Works**

`perform_hybrid_fusion(results_a, results_b, k=0, weight_a=1.0, weight_b=0.65)` takes the two `RunGenerator` result lists (from `run_single_seed()` or `run_official_ecf()`) and merges them with **Weighted Reciprocal Rank Fusion**: for each topic, every folder's score is `weight * (1 / (k + rank))` summed across the two rankings, then re-sorted.

1.  **Ranker A (Content-Based):** A normal `RunGenerator`, typically using `['title', 'ocr', 'folderlabel', 'summary']` and one or more retrieval models, optionally with expansion.
2.  **Ranker B (Metadata-Based):** A `RunGenerator` with `searching_fields=[['folderlabel']]` and `all_folders_folder_label=True` — a pure folder-metadata ranker over the whole collection.
3.  **Fusion:** The two result lists are combined via `perform_hybrid_fusion`, with weight `1.0` for A and `0.65` for B by default.

**2. Usage Guide**

`hybrid_models.py`'s own `run_hybrid_experiment()` function is a standalone example (not used by `run_new_experiments.py`, which builds its own generator pairs via `make_allfl_bm25()` / `make_allfl_colbert()`). To adapt it, edit the two `RunGenerator` configs and the `run_folder_name`:

```python
# Configuration A: The "Deep Diver" (Document Content)
gen_A = RunGenerator(
    searching_fields=[['title', 'ocr', 'folderlabel', 'summary']],
    query_fields=['TD'],
    models=['bm25', 'embeddings', 'colbert'],
    expansion=['similar_snc'],
    rrf_input='docs',
    expansion_ceiling_k=1,
)

# Configuration B: The "Overviewer" (Folder Metadata, ALLFL)
gen_B = RunGenerator(
    searching_fields=[['folderlabel']],
    query_fields=['TD'],
    models=['colbert'],
    expansion=[],
    all_folders_folder_label=True,
)
```

Update `run_folder_name` (used as the output directory under `all_runs/`) to something descriptive, then run it directly from `src/`:

```bash
python hybrid_models.py
```

The results are saved and evaluated automatically (30-seed loop, aggregated at the end), ready for inspection in the Visualizer.

### 8. BM25F Field-Weight Tuning - `tuning_bm25/bm25_tuning.py`

`src/tuning_bm25/bm25_tuning.py` is the script used to derive the tuned BM25F field weights hardcoded in `BM_25_FIELD_WEIGHTS` (`src/models.py`). It indexes the full collection once (via `ECF_ALL_TRAINING_SET.json`, fields `title`/`folderlabel`/`summary`), then performs a grid search over `(w, c)` pairs per field (see `PARAM_GRID` at the top of the file), evaluating each combination's nDCG@10 against the folder QRELs.

Run it from `src/tuning_bm25/`:

```bash
cd src/tuning_bm25
python bm25_tuning.py
```

It prints the top-5 configurations and writes the full grid to `bm25f_tuning_results.csv` in the current directory. Note it builds its own Terrier index at `./tuning_index` (relative to the current directory) and does not use the OCR field (commented out of `PARAM_GRID` to keep the grid size manageable).

### 9. Wilcoxon Significance Testing

`src/stats_test/wilcoxon_test.ipynb` (backed by `ExperimentAnalyzer` in `src/stats_test/utils_wilcoxon_test.py`) performs pairwise statistical significance testing between two experiment runs, using the **Wilcoxon Signed-Rank Test** across the 30 random-seed trials. Usage inside the notebook:

```python
from utils_wilcoxon_test import ExperimentAnalyzer

analyzer = ExperimentAnalyzer(base_runs_path="../../all_runs")
data_a = analyzer.load_run("U5.TD-.W.TOFS.-.-")
data_b = analyzer.load_run("U5.TD-.B.TOFS.-.-")

analyzer.compare(data_a, data_b, name_a="W (RRF)", name_b="B (BM25F)")               # global comparison
analyzer.compare(data_a, data_b, name_a="W", name_b="B", topic_id="T18Eval-00001")   # single-topic deep dive
analyzer.scan_significant_topics(data_a, data_b, name_a="W", name_b="B")             # scan every topic for p < 0.05
```

* `compare()` prints win/loss counts and the Wilcoxon p-value, plots a histogram of per-seed score differences, and displays a signed-rank breakdown table (`display()` calls require a Jupyter environment).
* `scan_significant_topics()` iterates all 45 topics and reports only the ones where the difference is statistically significant (`p < 0.05`).

This is the same test exposed interactively (without the plots/tables) in the Visualizer's **Two-Experiment Viewer**, see below.

### 10. Four-Way ANOVA

`src/stats_test/anova/` implements a four-way ANOVA (topic x training-set/seed x query-field x a 4th chosen factor) over the six-part run-name convention described in [section 6](#6-running-the-full-experiment-matrix---run_new_experimentspy).

1. **`build_grid.py`** parses run folder names directly out of `all_runs/`, finds which factor combinations form a *complete, balanced* rectangle (every topic x training-set x query-field x factor-level cell present), and emits the long-format CSV the ANOVA script expects:

```bash
cd src/stats_test/anova
# see what complete factor grids are already available in all_runs/
python build_grid.py inventory --trec-eval-dir ../../../all_runs

# emit the ANOVA input CSV for a chosen grid (example: vary Scoring, holding
# sample/fields/expansion/label-fusion fixed)
python build_grid.py emit --trec-eval-dir ../../../all_runs \
    --factor4 scoring --fix sample=U5 fields=TOFS prop=- labels=- \
    --out scores.csv
```

2. **`fit_anova.py`** fits the model in closed form (exact for a balanced, one-observation-per-cell design) and writes the ANOVA table, Tukey pairwise comparisons, and diagnostic plots:

```bash
python fit_anova.py --input scores.csv --outdir results/ --factor4-label "Scoring"
```

Precomputed results from a prior run of this pipeline are already checked in under `src/stats_test/anova/results/` (ANOVA tables, Tukey pairwise CSVs, and diagnostic/marginal-means plots) and `src/stats_test/anova/scores.csv`.

---

## SUSHI Visualizer Web Application ("SUSHI BAR")

The **Visualizer** ("SUSHI BAR", `web_app/app_sushi.py`) is the primary interface for exploring the dataset and analyzing experiment results. It is a Streamlit application with six pages, selected from a sidebar radio (`main()` in `app_sushi.py`).

It is designed to bridge the gap between raw metric/metadata files and actionable insights: dataset statistics, per-topic relevance browsing, training-set coverage analysis, and experiment benchmarking with confidence intervals and significance testing.

### Application Pages

**📦 Collection Viewer** (`run_data_overview_ui`) — explores the SUSHI collection itself (boxes/folders/documents, independent of any experiment): overview KPIs (box/folder/document counts, distinct SNC codes), documents-per-folder and folders-per-box histograms, SNC distribution broken down at the 1/2/3-level granularities (top/bottom-10 tables, histograms, top-40 bar charts, full tables), an "SNC Deep Dive" to browse all folders under one SNC code, and document browsing by SNC or by individual folder.

**🔍 Task Viewer** (`run_topic_viewer_ui`) — browse the 45 evaluation topics (Title/Description/Narrative), then drill into the hierarchical **Box → Folder → Document** tree of ground-truth relevant items for that topic, with star ratings for relevance grade (⭐⭐⭐ = Grade 3 Highly Relevant, ⭐ = Grade 1 Relevant).

**🧪 Training Set Viewer** (`run_ecf_inspector_ui`) — inspects what a given Experiment Control File (ECF)/Training Set makes "digitized": documents/folders covered, coverage by SNC and by box, coverage cross-referenced against the 45-topic QRELs (what % of relevant folders per topic are even reachable), and side-by-side comparison of two Training Sets. It reads directly from the pre-generated ECF files under `ecf/random_generated/` (`ECF_RANDOM_*.json` for uniform sampling, `ECF_UNEVEN_Random_Seed_*.json` for skewed sampling, `ECF_ALL_TRAINING_SET.json` for the oracle) — see the note in [section 1](#1-systemfile-architecture) above about these not being regenerated by a normal `RunGenerator` run.

**🔬 Single Experiment Viewer** (`run_single_experiment_ui`) — pick one run folder from `all_runs/` to see its global mean nDCG@5 ± 95% margin, a dumbbell chart of per-topic nDCG@5 with confidence intervals, a box plot of per-topic score variance across the 30 seeds, and — if a `run.txt` TREC run file has been placed manually inside that run's folder — a per-topic Retrieval Analysis table of the actual top-N retrieved folders against their QRELs grade.

**⚔️ Two-Experiment Viewer** (`run_two_experiment_ui`) — direct side-by-side comparison of any two run folders: global nDCG@5 delta, a **Wilcoxon signed-rank test** (paired on shared seeds, computed by `run_wilcoxon_test()` in `utils_experiments_viz.py` — no notebook needed) reporting the p-value and winner, an overlay dumbbell chart, and a Topic Separation breakdown categorizing every topic as Better (≥ +10%), About Equal (within ±10%), or Worse (≤ -10%) for Run A vs Run B.

**📖 How-To Guide** (`run_howto_ui`) — in-app reference covering the SUSHI task, the Box/Folder/Document hierarchy, SNC codes, Training Sets, nDCG@5, relevance grades, and a page-by-page tour. Note: its "Experiment Naming Conventions" section describes an older 5-part folder-naming scheme and predates the 6-part convention actually used by `run_new_experiments.py` (see [section 6](#6-running-the-full-experiment-matrix---run_new_experimentspy) of the Experiment Running guide above for the current one).

### Adding a New Experiment to the Visualizer

The Visualizer does not hardcode run names; it scans `all_runs/` recursively (`get_run_folder_info_map()` in `web_app/utils_experiments_viz.py`) and lists any directory containing at least one of these files:

```text
topics_mean_margin.json
model_overall_stats.json
all_documents_model_overall_stats.json
AllDocuments_TopicsFolderMetrics.json
topics_values.json
folder_overall_stats.json
```

In practice, running any experiment through `RunGenerator` / `run_new_experiments.py` / `hybrid_models.py` (see the [SUSHI Experiment Running](#sushi-experiment-running) section above) already produces these files under `all_runs/<run_folder_name>/`, so a new run just needs to finish before it shows up — no manual wiring is required. A few things to keep in mind:

- **Folder naming**: the app doesn't strictly require any particular naming scheme to *discover* a run, but grouping/sorting/labeling across pages assumes the same dotted convention already used throughout `all_runs/` (see [section 6](#6-running-the-full-experiment-matrix---run_new_experimentspy)). Keep new run folders consistent with that convention unless you're prepared for cosmetic mislabeling in the charts.
- **Retrieval Analysis (optional)**: the Single Experiment Viewer's per-topic retrieval drill-down only appears if you manually copy/rename a TREC-format run file to `run.txt` inside the run's folder (the Evaluator only ever writes the shared `results/RunResults.tsv`, which gets overwritten by the next run — copy the one you want to inspect before running anything else).
- **Model colors**: chart colors come from the `COLOR_MAP` dict in `web_app/utils_experiments_viz.py` (falling back to a deterministic hash-based palette for anything not listed). To pin a specific color for a new ranker/model, add an entry there keyed by however that run's model/ranker token appears (e.g. the single-letter ranker code, or a hyphenated model-name string).

### How to Run

1. **Install Python**: [https://www.python.org](https://www.python.org).
2. **Install the web app's dependencies** (a much lighter set than the full experiment pipeline's `requirements.txt`): run ```pip install -r requirements_webapp.txt``` (Streamlit, pandas, numpy, altair, plotly, streamlit-scroll-to-top). It is recommended to use a [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html) or a [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) env.
3. **Download all necessary files**: make sure to follow all the steps in the **Repository Setup** section of this README file (the Collection Viewer and Task Viewer need `data/` and `qrels/` populated; the Training Set Viewer needs `ecf/random_generated/` populated).
4. **Have at least one finished experiment** under `all_runs/` if you want to use the Single/Two-Experiment Viewers (see [SUSHI Experiment Running](#sushi-experiment-running) above) — this repository already ships with a full matrix of pre-computed runs.
5. **Run the Streamlit application** from the `web_app/` folder and open it in the browser:

```bash
cd web_app
streamlit run app_sushi.py
```