# [SUSHI](https://sites.google.com/view/ntcir-sushi-task/) Research

[Documentation](https://victorleaoo.github.io/SUSHI_Information_Retrieval_Archives/)

## Web Service SUSHI Data and Topic Visualizer

It was created a streamlit application that shows easier the metadata from folders and items. It also allows the visualization of the PDFs files.

For the topics matter, there is a quick visualization for the topics and what are the highly relevant and relevant boxes, folders and documents for them, making it simple to understand what are the expectations for a topic.

**How to use**:

1. Install the requirements: ```pip install requirements.txt```
2. Run the app: ```streamlit run app.py```
3. Open in the browser: ```http://localhost:8501```

## Project Structure & Required Files

To run the application correctly, you must reproduce the data structure locally, as large or sensitive files are not versioned in this repository (they are git-ignored).

```
├── data/                           # ⚠️ Create folder and add files manually
│   ├── folders_metadata/
│   │   └── FoldersV1.2.json        # ⚠️ Download Folder metadata
│   ├── items_metadata/
│   │   └── itemsV1.2.json          # ⚠️ Download Document metadata
│   └── raw/                        # ⚠️ Download and Place the raw Box/Folder structure with PDFs here
├── docs/
├── ecf/                            
├── qrels/                          # ⚠️ Add relevance judgment files here
│   ├── dry-run-qrels/              # ⚠️ Create folder and add files manually
│   │   ├── Ntcir18SushiDryRunBoxQrelsV1.1.tsv # ⚠️ Download
│   │   └── Ntcir18SushiDryRunFolderQrelsV1.1.tsv # ⚠️ Download
│   └── formal-run-qrels/           # ⚠️ Create folder and add files manually
│       ├── formal-box-qrel.txt # ⚠️ Download
│       ├── formal-document-qrel.txt # ⚠️ Download
│       └── formal-folder-qrel.txt # ⚠️ Download
├── src/
├── venv/                           # Virtual Environment (Create locally)
├── .gitignore
├── app.py                          # Main Streamlit application
├── mkdocs.yml
├── README.md
└── requirements.txt                # Python dependencies
```

All files can be found at the [SUSHI Test Collection](https://sites.google.com/view/ntcir-sushi-task/test-collection):

- [FoldersV1.2.json](https://drive.google.com/file/d/1U6BCx1_MWsymny_hOhurlfUapFa8HsTC/view?usp=sharing)
- [itemsV1.2.json](https://drive.google.com/file/d/1c_hpR_lgdGeskXaNQTdCS7nO9s1R2NOb/view?usp=share_link)
- [raw](https://drive.google.com/file/d/1hA5FW0cNloi20coLlGvnv5wMap8ZN8YL/view?usp=sharing)
- [Ntcir18SushiDryRunBoxQrelsV1.1.tsv](https://drive.google.com/file/d/1rVwHYOtY-PpG-RUo44H7lY9PmHNwbuL0/view?usp=sharing)
- [Ntcir18SushiDryRunFolderQrelsV1.1.tsv](https://drive.google.com/file/d/1Q6553xByDHcxhqMgrsNDT5O_a9f3VB9x/view?usp=sharing)
- [formal-box-qrel.txt + formal-document-qrel.txt + formal-folder-qrel.txt](https://drive.google.com/file/d/16vfBPyykpRHPbEfwGIVry5CAK7j-VzqR/view?usp=share_link)

## What is next?

- Test the [ir_explain](https://github.com/souravsaha/ir_explain) library for pointwise solutions (LIRME and EXS) for initial works.