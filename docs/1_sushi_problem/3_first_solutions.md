# First Solutions

This section covers the first solutions created for the problem that inspired SUSHI.

## [Known by the Company it Keeps: Proximity-Based Indexing for Physical Content in Archival Repositories](https://terpconnect.umd.edu/~oard/pdf/tpdl23.pdf)

### Hypothesis

Concept of **homophily**: the tendency to form strong social connections with people who share one’s defining characteristics.

- Thesis in this paper is that a form of **homophily is to be expected among the content found in archival repositories**. Archivists respect the **original order of archival content** when performing arrangement and description.
    * Respecting the original order of those records can help to **preserve the evidence of the creator’s activities**.
    * Makes it possible to **open collections for research use with a minimum of work** on the archivist’s part.
- Hypothesize that if we **know something about the content of some records in some archival unit** (e.g., folder, box, series, or repository) then we can make some plausible **inferences about where certain other records** that we have not yet seen might be found

### Data

- **Subject-Numeric Files**:
    * **1963-1973** records on paper;
    * **Primary subject** code (e.g., POL for Political Affairs & Relations)
    * **Second-Level** category: country;
    * **Third-level** category: numeric code specifying the primary subject (e.g., for POL, 27-12 is war crimes).

- Brown University engaged in digitization or records of Brazilian politics.
    * **14k items from POL-Brazil**;
    * **36 boxes** digitized content available;

Example of the metadata for folders:

```
POL 2-3 BRAZ 01/01/1967
POL 5 BRAZ 01/01/1967
POL 6 BRAZ 01/01/1967
```

### Experiments

When finding undigitized content is the goal, **all that a user of NARA’s archive would have is folder labels**. They would need to request every box containing any folder labeled with with a subject-numeric code and date related to their search goal.
- **Solution**: recommend to a user of the archive what box they should look.
    * **OCR from few documents** of each box to **create an index** to be used to search.
    * Query is some document titles (not used in the index creation).
    * **Ranking the boxes**: BM-25 with Porter stemmer.

- **Results**:
    * Randomly: **2.9%** right for the correct box;
    * OCR words of the first page: 
        - **27.9%** right for the correct box.
        - **40.4%** right for the correct box first or second guess.

The results shows that it is possible to work on the solution for the problem of finding relevant collections with not all materials indexed.

## [Searching for Physical Documents in Archival Repositories](https://terpconnect.umd.edu/~oard/pdf/sigir24edgeformer.pdf)

### Hypothesis

Based on the manual work searchers must have to find materials that is not yet digital, it is proposed to **build graphs from known content** and learned relationships and, from a user's query, **identify boxes with similar graph embeddings**.

It tries to evolve the aforementioned work by **solving the limitation of lexical matching only** and unifying OCR text and metadata for the ranking.

Also, extend to **uneven sampling**.

### Data

It creates two collections:
1. **Even sampling**: 5 documents per box;
2. **Uneven sampling**: vary the number of documents per box.

### Experiments

- **Solution**: The experiments used Graph-NeuralNetworks GNN (Edgeformers). Model **document** as one node type and **boxes** as the other, the **edge** being the **text** and the **category** being the **box**.
    * Given the edge and nodes, it tries to **predict the category (box)**. The model learns to create embeddings (vectors) for each category.
    * When the user inputs a **query**, it **creates the embedding** and look for the **nearest category**.

- **Results**:
    * When added other metadata to the graph, better results.
    * Results better then BM-25 from before.
        - Depends on the topic.

It also validates the problem solution and it shows that using embeddings for semantic may bring interesting results.