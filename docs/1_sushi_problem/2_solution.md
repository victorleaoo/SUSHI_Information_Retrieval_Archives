# The solution proposed by SUSHI

With the [problem addressed](./1_index.md) in mind, its possible to get to a proposed solution.

## Solution

The proposed solution is:

- Focused on **physical media** (e.g., paper or microfilm) and that **haven't been digitized or described individually** [1];
- Support the development of automated systems that can **learn from few digitized examples or other metadata at higher levels (folder/box)** [1];
    * From the learn above: **suggest where in a collection** a searcher might most productively look [1].

What SUSHI gives to help the solution development:

- **Test collections** that model the real problem;
- Insightful and affordable **evaluation**.

## Goals and Benefits

The goal is to **reduce the time and expense** of finding materials in an archive [1].

The goal of SUSHI Subtask A is to **build and evaluate search technology that can help the user** of an archive (e.g., a historian, a lawyer, or a private citizen investigating their family history) find things for which **no digital copy** exists and **no item-level metadata** has been created.

## The Folder Ranking Task

The main task is Folder Ranking:

- **Input**:
    * Given a **query** and an unsorted list of all folders in the collection [1];
    * Given data:
        - **Metadata** describing each of those **folders**;
        - Together with a **sparse sample of digitized documents**;
        - **Document-level metadata from some documents** in some of those folders.
- Output: 
    - A ranked list of folders a searcher might most want to see [1].

<figure markdown="span">
  ![Folder Ranking Task](../assets/folder_ranking.png){ width="600" }
  <figcaption>Folder Ranking Task [1]</figcaption>
</figure>

## References

> 1. [Searching Unseen Sources for Historical Information: Evaluation Design for the NTCIR-18 SUSHI Pilot Task.](https://terpconnect.umd.edu/~oard/pdf/emtcir24.pdf)