# Next Steps

In this markdown file, there's a list of functionalities to be added to the built systems.

## Data manipulation

The list of expected updates to be made at the "Data manipulation" system, which involves the folder label metadata creation:

1. Generate different possible variants of folder label medatada expansions.
    - Currently, it is only using the "with scope notes and with truncation" option.
    - Generate a folder metadata file that has all possible version fo folder label text.
    - Show in the SUSHI Visualizer.
2. Create code that generate uneven sampling ECF too.
    - The random folder selection is the same as before.

## SUSHI Experiment Runner

The list of expected updates to be made at the "SUSHI Experiment Runner" system is:

1. Add the configuration and running functionality for embeddings.
    - Make it possible for the user to choose to run with embeddings, instead of BM25F method.
    - Keep the same workflow for the other configurations and results stats.
    - Make it work for documents indexing fields (summary and titles), the folder of the highest similarity fields documment should be the one choosen for the ranking list.
2. Test runs using different combinations of indexing fields, such as "TO", "TOF".
3. Check the result confidence interval generator, since there are some cases in which it fails:
    - Setting random_seed_list = [100] in the random run generator, but I got a scipy error in the calculation of upper and lower bounds. Changing it to random_seed_list = [100, 100] doesn’t fix it.
4. Make possible to change the weights during the setup to test other BM25F weights setup.
5. Run code with uneven sampling.

## SUSHI Visualizer

The list of expected updates to be made at the "SUSHI Visualizer" system is:

1. Show how many runs were averaged for the results.
2. Adapt the Experiment Visualizer, in order to automatically check the folders and the different indexing fields combinations to add as filter.
3. Adapt the filter construction to work properly for the embeddings:
    - Does not properly load TD queries for F embeddings (it assumes only T is there, but TD is there too).
4. Get the right results from SUSHI Submissions from the topics to add for each filter, too (red dots).
5. Understand the necessary changes to: 
    - """The topic browser metadata doesn't show the folder metadata that is being used to index items, which requires some awkward gymnastics from the user to write down the folder number for a document, switch to folder metadata view, and then look at the metadata for that folder.  This design choice respects the actual metadata structure, but it would be more useful to follow that link and show the associated folder metadata in the browser (in the same way the OCR data is shown at the bottom)."""
6. Select which runs and how to generate correctly the red dots data (baseline), explain that it gets the average nDCG@5 from all runs, independently from filters, that's why it gets fixed.
    - Which selections from SUSHI Submissions will be made to be shown as red dots? **Terrier Baseline sent in the email?**