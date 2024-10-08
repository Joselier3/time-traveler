import os
import argparse
from dotenv import load_dotenv
from typing import List
from datetime import datetime, timedelta
import json
from pathlib import Path

from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.vectorstores.timescalevector import TimescaleVector
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.docstore.document import Document
from timescale_vector import client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.output_parsers import StrOutputParser

from langchain_core.retrievers import BaseRetriever
from selfquery import selfQuery

load_dotenv(".env.development.local")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]
JSON_STATEMENTS_DIR = Path("json_statements")
COLLECTION_NAME = "statements"

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--builddb", action="store_true", help="Builds a Timescale database from scratch with the specified json data")
ARGS = parser.parse_args()

class CustomTimescaleRetriever(BaseRetriever):
    db: TimescaleVector
    k: int = 4
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        queries = selfQuery(query)
        
        rstart_time = datetime.now()
        print(f"Realizando {len(queries)} busquedas...")
        docs_retrieved = []
        for queryObj in queries:
            docs = db.similarity_search(queryObj["query"], start_date=queryObj["start_date"], end_date=queryObj["end_date"])
            docs_retrieved.append({
                "title": queryObj["query"],
                "docs": docs
            })

        rend_time = datetime.now()
        rdelta = rend_time - rstart_time
        print(f"({int(rdelta.total_seconds()*1000)} ms)")
        return docs_retrieved

def extract_metadata(record: dict, metadata: dict) -> dict:
    year_dt_str = f"{record['years']['year_1']}-12-31 00:00:00"
    year_datetime = datetime.strptime(year_dt_str, "%Y-%m-%d %H:%M:%S")
    metadata["year"] = year_datetime.strftime(r"%Y-%m-%d %H:%M:%S")

    return metadata

def getJSONDocs():
    json_statements_filepaths = JSON_STATEMENTS_DIR.glob("*.json")

    documents = []
    for filepath in json_statements_filepaths:
        loader = JSONLoader(
            file_path=filepath,
            jq_schema='.',
            text_content=False,
            metadata_func=extract_metadata
        )
        docs = loader.load()
        page_content = json.loads(docs[0].page_content)
        metadata = docs[0].metadata

        splitter = RecursiveJsonSplitter(max_chunk_size=1000)
        split_docs = splitter.create_documents(texts=[page_content])

        for i, chunk in enumerate(split_docs):
            chunk_metadata = metadata.copy()
            chunk_metadata["seq_num"] = i + 1
            chunk_doc = Document(
                page_content=chunk.page_content,
                metadata=chunk_metadata
            )
            documents.append(chunk_doc)

    for d in documents:
        year_datetime = datetime.strptime(d.metadata['year'], "%Y-%m-%d %H:%M:%S")
        new_id = str(client.uuid_from_time(year_datetime))
        d.metadata["id"] = new_id

    return documents

def buildTimescaleDB(new_documents: list[Document]):
    embeddings = OpenAIEmbeddings()

    db = TimescaleVector.from_documents(
        embedding=embeddings,
        ids=[doc.metadata["id"] for doc in new_documents],
        documents=new_documents,
        collection_name=COLLECTION_NAME,
        service_url=SERVICE_URL,
        time_partition_interval=timedelta(days=7),
    )

    return db

def format_docs(docGroups):
    finalFormat = ""
    for docGroup in docGroups:
        title = docGroup["title"]
        doc_list = "\n".join(doc.page_content for doc in docGroup["docs"])
        finalFormat = f"{finalFormat}\n\n{title}{doc_list}"

    return finalFormat

def buildRAG(db: TimescaleVector):
    retriever = CustomTimescaleRetriever(db=db) 
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    template = """Responde la siguiente pregunta sobre los estados financieros de Citibank con el siguiente contexto:
    {context}

    Pregunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__=="__main__":
    if(ARGS.builddb):
        print("Construyendo base de datos...")
        docs_to_be_inserted = getJSONDocs()
        db = buildTimescaleDB(docs_to_be_inserted)
    else:
        print("Inicializando base de datos... ")
        tdb_start_time = datetime.now()

        embeddings = OpenAIEmbeddings()
        db = TimescaleVector(
            collection_name=COLLECTION_NAME,
            service_url=SERVICE_URL,
            embedding=embeddings,
        )

        tdb_end_time = datetime.now()
        tdb_delta = tdb_end_time - tdb_start_time
        print(f"({int(tdb_delta.total_seconds() * 1000)} ms)")

    rag_chain = buildRAG(db)
    
    print("Hola! Soy Max, tu asistente personal de datos bancarios.")
    print("Escribe \'quit\' para terminar la sesion.\n")
    print("Escribe tu pregunta aca abajo:")

    question = ""
    while True:
        question = input(">>> ")

        if question=='quit':
            break
        elif question=='':
            print("Por favor, haz una pregunta")
        else:
            print(rag_chain.invoke(question))