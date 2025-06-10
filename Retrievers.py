from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import TFIDFRetriever
from DataLoader import LoadContext
import warnings  
warnings.filterwarnings("ignore") 
import os 
import pandas as pd

documents = LoadContext("Data/fiqa/corpus.jsonl", "Data/PreparedData/EmbeddingRetriever.csv", "Data/fiqa/queries.jsonl", "Data/fiqa/qrels/train.tsv", 100, save_df=False) 

def GenerateEmbeddingRetriever():
    persist_directory = "Embeddings/chroma_db" 
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(persist_directory) and os.listdir(persist_directory):  
        print("Returning existing embedding retriever...")  
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        return db

    print("Generating embedding retriever...")
    
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_directory)
    db.persist()
    print("Embedding retriever generated and saved.")
    return db

def GenerateTFIDFRetriever():  
    return TFIDFRetriever.from_documents(documents)

def GenerateHeuristicRetriever():  
    df_docs = pd.read_csv("Data/PreparedData/HeuristicRetriever.csv")  
    return df_docs  
    