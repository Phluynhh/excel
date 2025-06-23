import os
from pymongo import MongoClient
from pyvi import ViTokenizer
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
collection = client["rag_db"]["excel"]

# Vietnamese stopwords (danh sách cơ bản, bạn có thể mở rộng)
vietnamese_stopwords = set([
    "và", "là", "của", "trong", "được", "có", "không", "tại", "với", "một",
    "các", "những", "đã", "đang", "sẽ", "bởi", "từ", "cho", "này", "đó"
])

# Embedding model (phải giống với model khi lưu embedding)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)

class Retriever:
    def __init__(self):
        # Load documents from MongoDB
        self.docs = list(collection.find({}))
        self.texts = [doc.get("text", "") for doc in self.docs]
        self.metadatas = [doc.get("metadata", {}) for doc in self.docs]
        self.files = [meta.get("file", "unknown") for meta in self.metadatas]
        self.configs = [meta.get("config", {}) for meta in self.metadatas]

        # Lấy embedding từ MongoDB nếu có, nếu không thì None
        self.embeddings = []
        for doc in self.docs:
            emb = doc.get("embedding")
            if emb is not None:
                self.embeddings.append(np.array(emb, dtype=np.float32))
            else:
                self.embeddings.append(None)

    def preprocess_query(self, query):
        """Preprocess query for Vietnamese using pyvi."""
        tokens = ViTokenizer.tokenize(query.lower()).split()
        return [token for token in tokens if token not in vietnamese_stopwords]

    def get_context(self, query, top_k=20, max_total_chars=50000, metadata_filter=None):
        """
        Retrieve context using only semantic search.
        Sử dụng embedding đã lưu trong MongoDB nếu có, nếu không thì tính động.
        metadata_filter: dict, e.g. {"file": "abc.xlsx"}
        """
        # Apply metadata filter
        filtered_indices = list(range(len(self.docs)))
        if metadata_filter:
            filtered_indices = [
                i for i, meta in enumerate(self.metadatas)
                if all(meta.get(k) == v for k, v in metadata_filter.items())
            ]
            if not filtered_indices:
                print("No documents match the metadata filter.")
                return ""

        # Get query embedding (phải dùng đúng model)
        query_embedding = embedding_model.embed_query(query)

        # Chuẩn bị embedding cho các document (ưu tiên lấy từ MongoDB)
        filtered_embeddings = []
        filtered_texts = []
        filtered_metas = []
        filtered_idxs = []
        for i in filtered_indices:
            emb = self.embeddings[i]
            if emb is not None:
                filtered_embeddings.append(emb)
                filtered_texts.append(self.texts[i])
                filtered_metas.append(self.metadatas[i])
                filtered_idxs.append(i)
            else:
                # Nếu không có embedding, tính động
                emb_dyn = embedding_model.embed_documents([self.texts[i]])[0]
                filtered_embeddings.append(np.array(emb_dyn, dtype=np.float32))
                filtered_texts.append(self.texts[i])
                filtered_metas.append(self.metadatas[i])
                filtered_idxs.append(i)

        if not filtered_embeddings:
            print("No embeddings found for filtered documents.")
            return ""

        # Tính cosine similarity
        similarities = cosine_similarity([query_embedding], filtered_embeddings)[0]

        # Sort by similarity
        scored_indices = sorted(
            zip(filtered_idxs, similarities, filtered_texts, filtered_metas),
            key=lambda x: x[1],
            reverse=True
        )

        context = ""
        total_chars = 0
        selected_chunks = []

        for i, score, text, meta in scored_indices[:top_k]:
            file_name = meta.get("file", "unknown")
            preview = f"### {file_name}\n{text}\n\n"
            if total_chars + len(text) > max_total_chars:
                preview = f"### {file_name}\n{text[:max_total_chars-total_chars]}\n\n"
            selected_chunks.append((score, preview))
            total_chars += len(text)
            if total_chars >= max_total_chars:
                break

        # Build context
        print("\n--- Top Semantic Chunks (embedding from MongoDB or dynamic) ---")
        for rank, (score, preview) in enumerate(selected_chunks):
            print(f"[{rank+1}] Score: {score:.4f} | Preview: {preview[:100]}...")

        context = "".join([preview for _, preview in selected_chunks])
        return context
