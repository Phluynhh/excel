from agents.query_analyzer import QueryAnalyzer
from agents.retriever import Retriever
from agents.response_generator import ResponseGenerator
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ dùng Gemini

class Orchestrator:
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.retriever = Retriever()

        self.model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.2
        )

        self.response_generator = ResponseGenerator(self.model)

    def handle_query(self, query, context=None):
        if not context:
            context = self.retriever.get_context(query)
        return self.response_generator.generate_response(query, context)

