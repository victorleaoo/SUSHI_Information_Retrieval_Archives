# Data collections

The data created and given by SUSHI is peculiar.

The selection of the sparsity can be:
    - **Even:** same number of documents per box.
        * This helps achieve **better control** over experimental conditions.
    - **Uneven:** different number of documents per box.
        * This emulates **real archives problems better** and help **characterize the additional challenges** of the problem.

## Full collection

All of the documents in the collection were created by the **State Department or the United States of America during the 1960’s and 1970’s**. The documents were selected for, and digitized by, the Brown University Libraries for use by scholars of **twentieth-century Brazilian history**. Some documents have metadata from both sources. The folder structure and the folder labels reflect the f**iling system used by the State Department when these were active records** [2].

The full collection is [1]:
    - **31,682 U.S. State Department documents** PDF files with unique identifier from;   
        * OCR and metadata (title, date, original folder label)
    - **1,337 folders** with unique identifier in;
    - **124 boxes** with unique identifier.

What makes the collection **easily judged** is that it is **fully digitized**, and that we have **topical metadata for every document** [1].

### Item-Level and Folder-Level Metadata

About **one-third of the documents were downloaded from NARA**, alone with NARA’s metadata for those documents. The **remaining documents** were obtained from **Brown**, along with Brown’s metadata for those documents [2].

In general, the **NARA metadata format** is somewhat **more consistent**, but somewhat **less expressive**, than that of the Brown metadata [2].

Since there documents with metadata from NARA and others only from Brown, it was created identifiers for the SUSHI task (Sushi Box, Sushi Folder and Sushi File). The Boxes from SUSHI ids are in a order of XNNNN, in which X is the collection and NNNN the sequential number. Folder and File are only unique ids, there's no order assigned [2].
- If starts with **N is from NARA only**;
- If starts with **M is from Brown only**.

#### NARA Metadata

There were too many metadata, so only the useful ones were extracted. NARA file names and NARA item (i.e., document) titles are both complex objects that could be further parsed [2].

Many NARA metadata types have both a unique identifier (called the National Archives Identifier, “NAID”) and a human-readable version [2].

#### Brown Metadata

First, Brown has no NAIDs, so all of the metadata is intended to be human-readable [2].

Second, Brown often describes the same thing a bit differently. For example, **NARA’s title for SUSHI document** S37333.pdf is “Telegram from State to Rio Concerning Considering U.S. Assistance to Brazil (3442): 1/9/1969” whereas **Brown’s title for the same document** is simply “Considering U.S. Assistance to Brazil” [2].

### Subject-Numeric Codes (SNC)

Method used to organize filing system. 

## Dry Run

### Test Collections

For the Dry Run, the sparse digitized sample is **even, including five documents per box**, with one document sampled from each of the five largest folders in each box [1].

### Topics creation and Relevance Judgments

**Randomly selection a document** that systems did not see at training time and using the **title** for the document as the query [1].
    - Treat any document with the **same title metadata as being relevant** [1].
    - Systems can't see those titles, since no training document (those with document-level metadata) is relevant [1].

Exact matching on document titles is useful for initial system development, but is a wak proxy for true human relevance judgements [1].

## Final Task

Plan to explore a mixture of even sampling (the same number of folders per box) and uneven sampling (with more samples from some boxes than from others) [1].

### Topics creation and Relevance Judgments

**Human assessors**, such as graduate students with a background in history or library science, are going to **make the judgements** [1]:

1. They will initially **create search topics** in the Title/Description/Narrative format;
2. They will check to see if **at least a few relevant documents exist**;
3. With a constructed system, they will **issue queries** (topics), **rank documents** based on one of several ways of indexing the collection (e.g., OCR-only, Title-only, or both) and **record the relevance judgment** for documents found relevant for the topics;
    * Later, it will be performed **Relevance Assessment**, in which more time will be allocated for more careful searching.

For folder-level judgment, it only aggregate document-level for the folder and uses the highest judgment.

## Data collections building challenges

The present approach is vulnerable to the common criticism of classic information retrieval test collections that they are not typically **designed to characterize cross-collection differences**. But in SUSHI we are seeking to model a real situation in **which different collections can have vastly different metadata structures**. And, of course, **real archival collections are not all equally amenable to OCR** — there are handwritten collections, photograph collections, collections written entirely in hieroglyphics or cuneiform, and (in one memorable case) a collection that consisted of nothing but x-ray film containing images [1].

- First, it must have content for which we **can create topics and for which we can perform relevance judgments** [1].
- Second, we must know **at least which box contained each item** (e.g., each document), and we would love to also **know which folder** in that box contained each item [1].
- Third, we really want to have **scanned images of all the documents** [1].
    * It takes a lot of time to get box access in archives.
        - So doing large numbers of relevance judgments by requesting and then examining paper doesn’t seem like a scaleable solution.
- It's hard to find large collections of digitized materials with good box and folder metadata [1].

## References

> 1. [Searching Unseen Sources for Historical Information: Evaluation Design for the NTCIR-18 SUSHI Pilot Task.](https://terpconnect.umd.edu/~oard/pdf/emtcir24.pdf)

> 2. [Guidelines for SUSHI Subtask A: Searching Undigitized Content](https://drive.google.com/file/d/1WNysJLaDIbS4rXS8rNG4hWNt05T71mBC/view)