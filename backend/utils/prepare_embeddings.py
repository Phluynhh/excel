import pandas as pd
from utils.embedding_store import store_embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)
semantic_chunker = SemanticChunker(embeddings=embedding_model)
recursive_chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def prepare_and_store_embeddings(file_path, config):
    sheet_name = config["sheetName"]
    header_row = int(config["headerRow"]) - 1
    data_row = int(config["dataRow"]) - 1
    indexed_columns = [col["name"] for col in config["columns"] if col["isIndexed"]]

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row, skiprows=data_row - header_row - 1)
    df = df[indexed_columns]

    text_chunks = []
    for _, row in df.iterrows():
        chunk = " | ".join([f"{col}: {row[col]}" for col in indexed_columns if pd.notna(row[col])])
        text_chunks.append(chunk)

    docs = recursive_chunker.create_documents(text_chunks)

    # Tính embedding cho từng chunk
    embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])

    # Gọi lưu embeddings (truyền thêm embeddings)
    store_embeddings(docs, file_path, config, embeddings)
