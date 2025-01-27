import streamlit as st
import pandas as pd
from pymongo import MongoClient
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


MONGO_URI = "mongodb://iRecon:AppleiRecon%2314210@rn3-irecont-lmdb08.rno.apple.com:10906/CCiRecon_I001_DEV0?authSource=CCiRecon_I001_DEV0"
DB_NAME = 'CCiRecon_I166_DEV0'
COLLECTION_NAME = 'iReconTxCollection'


COLUMNS = ['CreatedDate', 'WebOrderNumber', 'AcquirerRefNumber', 'AcquirerId', 'TransactionType', 'Amount', 'DocumentDate', 'ImportBatchID', 'match_res', 'match_updates']

@st.cache_data
def fetch_mongodb_data(limit=50000):
    client = None
    try:
        logging.info("Attempting to connect to MongoDB...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        client.server_info()
        logging.info("Successfully connected to MongoDB")
        
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        logging.info(f"Accessing database: {DB_NAME}, collection: {COLLECTION_NAME}")
        
        logging.info(f"Attempting to fetch {limit} documents...")
        cursor = collection.find({}, projection=dict.fromkeys(COLUMNS, 1)).limit(limit)
        
        df = pd.DataFrame(list(cursor))
        
        if df.empty:
            logging.warning("Query returned no results")
        else:
            logging.info(f"Successfully fetched {len(df)} documents")
        
        return df
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        return None
    
    finally:
        if client:
            client.close()
            logging.info("MongoDB connection closed")

@st.cache_resource
def load_embeddings_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_faiss_index(df: pd.DataFrame, model):
    texts = df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index, embeddings

def search_similar(query: str, index, embeddings: np.ndarray, model, df: pd.DataFrame, k: int = 5) -> List[dict]:
    query_vector = model.encode([query])[0]
    distances, indices = index.search(query_vector.reshape(1, -1).astype('float32'), k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'text': df.iloc[idx].to_dict(),
            'distance': distances[0][i]
        })
    return results

def main():
    st.title("RAG Chatbot with Local Data")

    
    df = fetch_mongodb_data()
    if df is None or df.empty:
        st.error("Failed to fetch data from MongoDB.")
        return

    
    model = load_embeddings_model()
    index, embeddings = create_faiss_index(df, model)

    
    st.subheader("Ask a question about the data:")
    user_input = st.text_input("Your question:")
    if user_input:
        results = search_similar(user_input, index, embeddings, model, df)
        
        st.subheader("Relevant Information:")
        for i, result in enumerate(results, 1):
            st.write(f"Result {i}:")
            st.json(result['text'])
            st.write(f"Similarity: {1 - result['distance']:.4f}")
            st.write("---")

if __name__ == "__main__":
    main()
    