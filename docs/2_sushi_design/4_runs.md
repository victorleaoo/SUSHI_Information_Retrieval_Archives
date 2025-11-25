# Experiment Runs and Submissions

## Data that can be used

The [data](./2_data.md) that can be used is:
- Systems may train using both the **digitized content** from the specified sample and any of the provided **item-level** metadata from that specified sample;
- Systems may also train using other **metadata (e.g., folder labels)** for the entire test collection;
- However, systems may **NOT train machine learning systems** on any of the digitized content or item-level metadata from **items outside the specified training sample** in any way. Folder-level can be used.

## SUSHI Runs

A SUSHI run is a set of ranked lists, one for each topic in the experiment control file.

All runs must use the T field in their queries.

### Experiment Control Files

The simulation of limited digitization in SUSHI Subtask A requires that the allowable training data be specified for each topic.

Experiment Control Files contain three components:
- **Experiment Name:** A unique human-readable identifier for the experiment control file;
- **Training Set:** A list of full file paths in the test collection, one for each allowable training document in this experiment;
- **Topics:** Specifications of what the system is asked to find (relevant folders with relevant documents). Each topic includes at least a query field containing a query string that is representative of what a searcher might type (TDN).
    * **TITLE**: A short query-like text string.
    * **DESCRIPTION**: A brief human-readable description of the topic, as a text string.
    * **NARRATIVE**: A human readable guide for relevance assessment, as a text string.

```
{
    "ExperimentName": "NTCIR-18 SUSHI Dry Run v1.0, 200 Queries, Uniform Sample, 5
    Documents Per Box",
    “ExperimentSets”: [
        {
            “TrainingDoucments”: [
                “N1234/N12345678/S12345.pdf”,
                “N1239/N12345667/S11213.pdf”,
                ...
                “N1034/N12344221/S14352.pdf”
                ],
            "Topics": {
                "T18-42001" : {
                    "TITLE": "Resignations of government ministers",
                    "DESCRIPTION": "Find folders containing documents addressing
                    the resignation of one or more ministers of any government",
                    "NARRATIVE": "These documents might be personal, official, or
                    public. For example, news reports speculating on the likelihood of, or
                    reasons for a resignation, would be relevant, as would resignation letters,
                    responses to resignation letters, or official messages describing a
                    resignation. Only resignations of government ministers are relevant;
                    resignations by members of the permanent civil service (or equivalent)
                    are not relevant."
                },
                ...
            }
        }
        ...
    ]
    ...
}
```

## Run Submission Format

Each run is to be submitted as one Tab-Separated Values (TSV) file with one line for each folder retrieved by the system.
- **TopicID**: A topic identifier;
- **SushiFolder**: The Sushi Folder identifier;
- **Rank**: The rank for the folder. Ranks are integers between 1 and 1,000, with Rank 1 indicating the most highly ranked folder;
- **Score**: The relevance score used by the system to perform the ranking. Larger values indicate that a folder is seen by the system as a better choice.
- **RunID**: A unique run identifier that includes no whitespace.

```
T18-00001 N23813037 1 36.3 KyuNLP-A1
T18-00001 N23813038 2 27.0 KyuNLP-A1
T81-00001 N23812863 3 26.5 KyuNLP-A1
T18-00001 N23812865 4 11.9 KyuNLP-A1
T18-00001 N23812867 5 11.9 KyuNLP-A1
...
T18-00001 N23812850 1000 0.0 KyuNLP-A1
T18-00003 M99990700 1 4.3 KyuNLP-A1
T18-00003 M99990497 2 0.2 KyuNLP-A1
```

## Dry Run

It models a process that participants may use for **formative evaluation of their own system**. It simplifies the task of first managing the complex ECF and metadata files.