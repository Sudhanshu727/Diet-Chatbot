# E:\Diet Chatbot\rag\retriever.py
from .knowledge_base import KnowledgeBase # <--- CORRECT: Relative import for sibling

# Placeholder for the actual retriever instance, initialized in main.py
rag_retriever = None

# Function to set the retriever instance after it's initialized in main.py
def set_rag_retriever(retriever_instance):
    global rag_retriever
    rag_retriever = retriever_instance

# You can also add a function to easily get the retriever elsewhere
def get_rag_retriever():
    if rag_retriever is None:
        raise RuntimeError("RAG retriever not initialized. Call set_rag_retriever from main.py first.")
    return rag_retriever