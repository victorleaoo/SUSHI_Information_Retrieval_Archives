# SUSHI Phase 0 — Data Analysis Dashboard

## Overview

A comprehensive Streamlit application for Phase 0 data analysis of the SUSHI archive collection (U.S. State Department records on Brazil, 1960s-1970s), covering **all checklist items** from `experiments.md`.

## How to Run

```bash
streamlit run web_app/app_data_analysis.py
```

Then open **http://localhost:8501** in your browser.

## Pages & Features

### 🏠 Overview

- Key collection metrics: 1,336 folders, 31,681 documents, 379 distinct SNCs
- Documents-per-folder and folders-per-box distributions
- Primary SNC code overview with folder & document counts

### 📊 0.1 SNC Distribution

- **3-Level SNC**: Full table, top-10 & bottom-10, histogram, scope note flags
- **2-Level SNC**: Grouped view (e.g., "POL 18" covers "POL 18-1", "POL 18-2")
- **1-Level (Primary)**: Treemap + bar chart
- **Documents per SNC**: Scatter plot of folder count vs document count, avg docs/folder

### 🔍 0.2 Digitization Coverage per SNC

- Select any Uniform ECF (30 available)
- Per-SNC coverage: Rich (≥50%), Moderate (20-49%), Poor (<20%)
- Pie chart and histogram of coverage categories
- Lists SNCs with >80% empty folders (where homophily won't help)

### 🤝 0.3 Homophily Feasibility Study

- Interactive threshold slider (D7 decision)
- SNC matching level selector: Exact, 2-Level, Both (D6 decision)
- Key stats: % of empty folders with ≥threshold neighbor docs
- Cumulative distribution chart
- Per-SNC breakdown table
- **Automatic D9 decision**: recommends Proceed/No based on 20% viability threshold

### 📝 0.4 Scope Note Coverage

- Counts folders with/without scope notes
- Primary SNC breakdown — which SNCs benefit from F3/F4 configurations
- Browse all scope note texts

### 📅 0.5 Date Ranges

- Start year distribution histogram
- Year × primary SNC stacked bar chart
- Flags folders with missing/anomalous dates ("Unknown" end dates)

### 🏷️ 0.6 Textual SNC Analysis

- Identifies folders with NULL/missing SNC codes (68 "Unknown" found)
- Label completeness: 1965 labels, 1963 labels
- Top-level SNC codes visualization
- Word frequency analysis of expanded SNC labels

### 📄 0.7 Document Analysis

- **Statistics**: OCR pages distribution, docs-per-folder histogram, top 30 title keywords
- **Browse by SNC**: Select any SNC, see its folders, expand to view documents with titles/summaries
- **Search**: Free-text search across all document titles

### 🔎 SNC Deep Dive

- Select any SNC code for a comprehensive 360° view:
  - Folder list with doc counts and scope note status
  - Browse documents within specific folders (titles + summaries)
  - **Keywords & Themes**: word frequency from titles and summaries separately
  - **Timeline**: documents per year + monthly heatmap
  - **Statistics**: document and folder stats side-by-side
  - Documents-per-folder bar chart

## Key Findings from the Data

| Metric                               | Value  |
| ------------------------------------ | ------ |
| Total Folders                        | 1,336  |
| Total Documents                      | 31,681 |
| Distinct SNCs                        | 379    |
| Primary SNC Codes                    | ~30    |
| Folders with Scope Notes             | 504    |
| Folders with SNC="Unknown"           | 68     |
| Avg Docs/Folder                      | ~23.7  |
| ECF Training Docs (Uniform, per ECF) | ~628   |
| Unique Folders in ECF                | ~562   |

### Important Findings

- There are a relevant number of folders with SNC == 'Unknown'. This is crucial, because if there are no documents, these folders cannot be found.
  * Solution: Use instead the 'label' field for searching and expanding.
- 561 folders have the parent SNC POL, which means that there are a lot of folders, and, thus, documents, related to some subjects.
  * Solution: it doesn't need to look for all the documents of some SNCs, just selected a subset from different folders. For example, if a SNC is going to be expanded and there are a lot of documents, select only a subset (2 random for each folders of the SNC) to use as an example for the prompt.
- It doesn't need to use only same snc for the homophily, if there is no document with same folder snc, it can go up (2-level, parent) until it finds documents to use.
- Scope note is very restricted. Use it when possible, but have in mind that some SNC have more information than others.
- There are a lot of folders with missing dates. Use it when possible, but have in mind that some folders lack information.
- **LOOK FOR LABEL PARENT EXPANDED OF SOME SNC**

## Technology Stack

- **Streamlit** 1.51 — interactive web framework
- **Plotly** 6.8 — interactive charts (dark theme)
- **Pandas** — data manipulation
- Python 3.10
