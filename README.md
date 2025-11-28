# [SUSHI](https://sites.google.com/view/ntcir-sushi-task/) Research

[Documentation](https://victorleaoo.github.io/SUSHI_Information_Retrieval_Archives/)

## Web Service SUSHI Data and Topic Visualizer

It was created a streamlit application that shows easier the metadata from folders and items. It also allows the visualization of the PDFs files.

For the topics matter, there is a quick visualization for the topics and what are the highly relevant and relevant boxes, folders and documents for them, making it simple to understand what are the expectations for a topic.

**How to use**:

1. Install the requirements: ```pip install requirements.txt```
2. Run the app: ```streamlit run app.py```
3. Open in the browser: ```http://localhost:8501```

## Next Steps

- Add the MULTIPLEX explanation (ir_explain and proposed work)

- Run formal-run with the same model from dry run (for each topic, save a file with the ranking + metrics per topic)
- see if there's a way to see index and bm25 choices
- add to webservice run topic and box checking and viz => with trainingdata

- Run quick exir model for results
- Document 