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

        # Chỉ lấy embedding đã lưu, bỏ qua doc chưa có embedding
        self.embeddings = []
        self.valid_indices = []
        for idx, doc in enumerate(self.docs):
            emb = doc.get("embedding")
            if emb is not None and self.files[idx].endswith(".xlsx"):
                self.embeddings.append(np.array(emb, dtype=np.float32))
                self.valid_indices.append(idx)
        # Lưu lại texts, metadatas, files tương ứng
        self.texts = [self.texts[i] for i in self.valid_indices]
        self.metadatas = [self.metadatas[i] for i in self.valid_indices]
        self.files = [self.files[i] for i in self.valid_indices]

    def preprocess_query(self, query):
        """Preprocess query for Vietnamese using pyvi."""
        tokens = ViTokenizer.tokenize(query.lower()).split()
        return [token for token in tokens if token not in vietnamese_stopwords]

    def get_context(self, query, top_k=50, max_total_chars=100000, file_name=None):
        """
        Retrieve context using only semantic search for Excel files.
        file_name: chỉ lấy context từ file Excel cụ thể (nếu có).
        """
        # Lọc theo file Excel nếu có
        filtered_indices = list(range(len(self.texts)))
        if file_name:
            filtered_indices = [
                i for i, meta in enumerate(self.metadatas)
                if meta.get("file", "").lower() == file_name.lower()
            ]
            if not filtered_indices:
                print("No documents match the file filter.")
                return ""

        # Lấy embedding cho query
        query_embedding = embedding_model.embed_query(query)

        # Lấy embedding và texts đã lọc
        filtered_embeddings = [self.embeddings[i] for i in filtered_indices]
        filtered_texts = [self.texts[i] for i in filtered_indices]
        filtered_metas = [self.metadatas[i] for i in filtered_indices]

        if not filtered_embeddings:
            print("No embeddings found for filtered documents.")
            return ""

        # Tính cosine similarity
        similarities = cosine_similarity([query_embedding], filtered_embeddings)[0]

        # Sắp xếp theo similarity giảm dần
        scored_indices = sorted(
            zip(filtered_indices, similarities, filtered_texts, filtered_metas),
            key=lambda x: x[1],
            reverse=True
        )

        context = ""
        total_chars = 0
        selected_chunks = []

        # Lấy nhiều chunk hơn, không cắt chunk giữa chừng
        for i, score, text, meta in scored_indices[:top_k]:
            file_name = meta.get("file", "unknown")
            preview = f"### {file_name}\n{text}\n\n"
            if total_chars + len(text) > max_total_chars:
                break  # Không thêm nữa nếu vượt quá giới hạn
            selected_chunks.append((score, preview))
            total_chars += len(text)

        # Build context
        print("\n--- Top Semantic Chunks (Excel only, embedding from MongoDB) ---")
        for rank, (score, preview) in enumerate(selected_chunks):
            print(f"[{rank+1}] Score: {score:.4f} | Preview: {preview[:100]}...")

        context = "".join([preview for _, preview in selected_chunks])
        return context
