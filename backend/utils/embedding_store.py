from pymongo import MongoClient
from datetime import datetime
from langchain.vectorstores import FAISS
import os

client = MongoClient("mongodb://localhost:27017")
db = client["rag_db"]
collection = db["excel"]

def store_embeddings(documents, file_path, config, embeddings=None):
    for idx, doc in enumerate(documents):
        record = {
            "text": doc.page_content,
            "metadata": {
                "file": os.path.basename(file_path),
                "config": config,
                "timestamp": datetime.utcnow()
            }
        }
        # Nếu là file Excel và có embedding thì lưu embedding
        if file_path.lower().endswith((".xls", ".xlsx")) and embeddings is not None:
            record["embedding"] = embeddings[idx]
        collection.insert_one(record)
