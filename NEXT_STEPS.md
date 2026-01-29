# Next Steps

In this markdown file, there's a list of functionalities to be added to the built systems.

## Data manipulation

**Updates already made:**

- SNCLabelTranslate file creates a new folder_metadata.json more flexible (all possible fields). Solves:
  - Generate different possible variants of folder label medatada expansions.
    * Currently, it is only using the "with scope notes and with truncation" option.
    * Generate a folder metadata file that has all possible version of folder label text (select which one for the run).

=> From new FoldersV1.3.json:
  - Change the name *stoppers* to *stopped/truncated*.

- Document how each one was created and the relevance in the README, in order for a new user to know which one to use.

**Urgent Updates:**
- Check and put the date of the scope note (1965 or 1963). 1965 -> look at POL 17 diff // 1963 -> look at POL 27 diff

## SUSHI Experiment Runner

**Updates already made:**

- ECF is one training documents set for all topics
- Implemented Embeddings and ColBERT to follow the same workflow as bm25
  - BM25 is pretty fast
  - Embeddings and Colbertwork with small texts like Title
    * With OCR, it was taking 1 minute per run
    * With Summary, it was taking 20sec per run
- Added the RRF workflow

=> Experiments Runs
Construct the best model to retrieve documents (answer the question of how the best model can work):
- **For all models (bm25, colbert, embeddings and their rff):**
  **Model 1 (Training Documents):** BEST MODEL: TFS_NE_TD_BM25-COLBERT
    1. Run only query: 
      - TD
    2. Run searching fields (with default BM25 params):
      - Each one of the fields independently
      - TOFS
      - TFS
      - TFO
  **Model 2 (Folder Label All Folders):** BEST MODEL: ALLFL_NE_TD_COLBERT (only parent label)
    1. Run Folder Label search in all folder training documents:
      - Use different folder labels constructions
- **Final First Version Model:**
  1. Combine **Best Model 1** and **Best Model 2** with RRF to get the final result.
  2. Test with query T.
- **Tuning to get to Final Model:**
  1. If BM25 is among the **Best Model 1**, tune using the Doug's approach (tune the params of each field and k1/b).
  2. Run the **Best Model 1** again with tuned params.

Construct and test expansion best on the scores of the **Final Model**:
- **Check if the expansion works as expected:**
  - Same Box AND Same SNC (must be both)
    => CHANGED THE CODE TO WORK WITH THE TECHNIQUES IN THE ARGS
  - If the expansion folders go up to the second rank 
    => FROM THE OLD CODE, IT WAS GOING UP TO 5TH RANK, NOW IT SELECTS UP TO WHERE
- **Run the best model with different Expansion Techniques to see how it performs**
- New way of evaluation the results of expansion:
  - Check how **many folders that have relevant documents (and were not scored before) got scored up by expansion**
    * Count how many folders from expansion got to the top 5 (not look only top nDCG@5)
  - Get expandable topics per ECF

Freeze -> Run -> Tune -> Score -> Expansion

Runs:
1. T/O/F/S (No Expansion)
  * All Models: BM25 / COLBERT / EMBEDDINGS / BM25+COLBERT / BM25+EMBEDDINGS
2. TOF / TFS / TOFS (No Expansion)
  * All Models: BM25 / COLBERT / EMBEDDINGS / BM25+COLBERT / BM25+EMBEDDINGS
  * BM25 Tuned for TFS and TOFS **CHANGE NAME**  
3. All Folder Labels
  * All Models: BM25 / COLBERT / EMBEDDINGS / BM25+COLBERT / BM25+EMBEDDINGS 
  * **Parent Only** / Scope Stopped Only / Parent + Scope Stopped
4. Hybrid RRF => Best(1 OR 2 = **TOFS_NEX_TD_BM25-EMBEDDINGS-COLBERT-TUNED**) AND Best(3 = **ALLFL_NEX_TD_COLBERT**)
5. For Best(HYBRID-TOFS-NEX_ALLFL-COLBERT_NE_TD_BM25-EMBEDDINGS-COLBERT-TUNED-WRRF)
  * Test different Expansion Methods
    - Different Combinations of Techniques
      -> Best Model (TOFS_NEX_TD_BM25-EMBEDDINGS-COLBERT-TUNED) with all techniques at rank 2
        -> Best Model was TOFS_SMS-2_TD_BM25-EMBEDDINGS-COLBERT-TUNED, so tried with rank 3 (not better)
    - Then tried the Best Model of All with this Expansion Technique
  
- Document how to Run everything on Windows (with no GPU)

**Urgent Updates:**

- Design Uneven Sampling experiments, but only run the Final Model.

## SUSHI Visualizer

**Updates already made:**

- Show how many runs were averaged for the results.
- Adapt the Experiment Visualizer, in order to automatically check the folders and the different indexing fields combinations to add as filter.
- In the data visualizer, add all possible metadata
- Document how it works to add a new model

**Urgent Updates:**

Document the code better.

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


---

tuning:

# About the Tuning Params:
# 1. The BM25F parameters are not a question about any particular ECF, so you don't want to start with the ECF that you will use in test -- a person making a BM25F system would not know the specific ECF's that would be user at test time.  
# 2. The second thing to say is that BM25 includes a document independence assumption, which means that we first compute a score for each document and then rank the documents. So the training mask does not change the scores of the selected documents.  

# **"Unfair Optimization:"**
# "I would suggest just doing the unfair optimization on the entire topic set first, and then circling back to see if we have time to add cross-validation at the end."
    # **Implementation:** This suggests a really simple design for BM25F parameter tuning in which you use no training mask (the all-documents condition).
    #   - Now you can easily optimize without cross-validation by just doing a grid search in the parameter space.
    #       * BM25F has 8 parameters for four fields, so you will probably want something more efficient than a grid search. 
    #   - And you'll have a lot of ties (because of the small number of relevant documents) so you want something that finds midpoints in the parameter space for tied maximal values -- this is a sort of maxent approach, but on quantized results (so its not really maxent).  
    #   - I suggest to you do some by-hand eploration of the parameter space (one dimension at a time) to see this quantization effect.  
    #   - Choose an evaluation measure to optimize for: nDCG@10 might be better than nDCG@5, since it would have fewer ties.

# **Cross-Validation:**
    # - To implement cross-validation, choose some test topics and optimize the BM25F parameters on the other topics. 
    # - Then whenever you run an actual experiment whenever you test on one of those test topics, just use the BM25 parameters that were learned on the other topic.  
