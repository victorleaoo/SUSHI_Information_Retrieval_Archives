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

- documentar a estrutura de pastas do repositorio pra saber o que baixar e como colocar a fim de replicar tudo que está aqui
- calculo de resultado do topico (pode ser simples, so se encontrou ou nao documentos relevantes)
    - colocar isto direto na run para ficar mais facil de ver
    - adicionar isto no webservice "topic score"
- see if there's a way to see and explain index and bm25 choices


- Run quick exir model for results (lib: ir_explain -> lirme)
- Document