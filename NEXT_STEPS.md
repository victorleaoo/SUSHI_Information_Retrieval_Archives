# Next Steps

In this markdown file, there's a list of functionalities to be added to the built systems.

## Data manipulation

**Updates already made:**

- SNCLabelTranslate file creates a new folder_metadata.json more flexible (all possible fields). Solves:
  - Generate different possible variants of folder label medatada expansions.
    * Currently, it is only using the "with scope notes and with truncation" option.
    * Generate a folder metadata file that has all possible version of folder label text (select which one for the run).

**Urgent Updates:**

- Change the name *stoppers* to *stopped/truncated*.
- Create one with *stop words* to call *stoppers*.
- Check and put the date of the scope note (1965 or 1963).
- Document how each one was created and the relevance in the README, in order for a new user to know which one to use.

## SUSHI Experiment Runner

**Updates already made:**

- ECF is one training documents set for all topics
- Implemented Embeddings and ColBERT to follow the same workflow as bm25
  - BM25 is pretty fast
  - Embeddings and Colbertwork with small texts like Title
    * With OCR, it was taking 1 minute per run
    * With Summary, it was taking 20sec per run
- Added the RRF workflow

**Urgent Updates:**

Freeze -> Run -> Tune -> Score -> Expansion

Construct the best model to retrieve documents:
**For all models (bm25, colbert, embeddings and their rff):**
  **Model 1 (Training Documents):**
    1. Run only query: 
      - TD
    2. Run searching fields (with default BM25 params): => Decide which is the best result
      - Each one of the fields independently
      - TOFS => Test with label_parent_expanded only and label_parent_expanded + scope_stoppers (FIRST)
      - TFS
      - TFO
  **Model 2 (Folder Label All Folders):**
    1. Run Folder Label search in all folder training documents:
      - Use different folder labels constructions

**Final First Version Model:**
  1. Combine **Best Model 1** and **Best Model 2** with RRF to get the final result.
  2. Test with query T.

**Tuning to get to Final Model:**
  1. If BM25 is among the **Best Model 1**, tune using the Doug's approach.
  2. Run the **Best Model 1** again with tuned params (tune the params of each field and k1/b from bm25).

==> Until here, answers the question of how the best model can work

Construct and test expansion best on the scores of the **Final Model**:
  1. Check if the expansion works as expected:
    - Same Box AND Same SNC (must be both)
    - If the expansion folders go up to the second rank
  2. Run the best model with different Expansion Techniques to see how it performs
    - *(not now)* Folders with not only 0 documents being inserted into the expansion list
  3. New way of evaluation the results of expansion:
    - Check how many folders that have relevant documents (and were not scored before) got scored up by expansion
      * Count how many folders from expansion got to the top 5 (not look only top nDCG@5)
    - Get expandable topics per ECF
  
- How to Run everything on Windows (with no GPU)

==> The experiments should go this far

- Design Uneven Sampling experiments, but only run the Final Model.
- *No need for cross-val*

## SUSHI Visualizer

**Urgent Updates:**

- Show how many runs were averaged for the results.
- Adapt the Experiment Visualizer, in order to automatically check the folders and the different indexing fields combinations to add as filter.
  * Here it needs to have all the filters from the different runs types
  * Adapt all the metrics and topics to work properly
  * There must be the results of BM25, Embeddings and ColBERT for every single run type
- In the data visualizer, add all possible metadata

**Ask to Doug:**

- Is there any need to add the results from the SUSHI Submissions? The results are not in the same way as we are working right now
  * Get the right results from SUSHI Submissions from the topics to add for each filter, too (red dots).
  * Which selections from SUSHI Submissions will be made to be shown as red dots?
    - Maybe only have a Terrier Baseline model to show for each run type?

**Future Work for the Paper:**

- Create a searching system and make it available on the internet

## Paper

**Updates already made:**

- Describe how OCR was done
- Update Figure 1, separating as two differen Figures:
  1. "Left part": where the collections are coming from.
     * Add the creation of the Four Fields (TOFS) and where they belong (F from folders and TOS from the documents)
  2. "Right part": the topic creation and the relevance judgment are different processes
     * Reacreate with the different processes
     * Show real examples of Folders and Documents IDs
     * Show that the topics feed the ECF, but the Relevance Judgments don't feed anything
     * Don't link topic and relevance (they are different processes)
- Create new Figure:
  * Show the "expansion" process and the importance
    - Two boxes left and right with similar theme (with documents) and box in the middle with no theme, but it tends to have the similar theme
    - How expansion is done

**Urgent Updates:**

- Change Document Title (T) to L (document Label), in order to not have the topic T confused

**After talking to Doug:**

- Fill the table with the experiments results and explain what is most relevant in the text
  - Storytelling

---

**MOST IMPORTANT**:

- Show the benefit of Expansion, in order to revolutionize IR field area
  * Show that it is possible to use other folders to score RELEVANTLY from a folder without document

**Doug's thought on Professor Thiago's idea:**

- The metadata is describe as normative (gramatical) and LLMs try to describe as natural humans, therefore, it can ends being unmatching