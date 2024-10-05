# Time Traveler: time-based RAG for banking Q&A

Time Traveler is a python package for a RAG assistant that searches as many documents as necessary in a Timescale Vector database to fulfill the needs of a question. It dynamically generates multiple queries to search different documents, depending on the context of the question. It uses Langchain to build the RAG chain and to make individual similarity searches to the database.

This project was inspired by [Langchain's Timescale Vector guide](https://python.langchain.com/docs/integrations/vectorstores/timescalevector/).

## Usage

1. Put your OpenAI API key and your timescale service url in the .env file.

2. Create an environment using `environment.yml`.
```
mamba env create --file environment.yml -n time-rag
mamba activate time-rag
```

3. Make a sample question to the rag chain.
```
python rag.py "Dame el total de activos de los ultimos 3 años"
```
Mark the `--builddb` option if you wish to build the Timescale Vector database for the first time.
