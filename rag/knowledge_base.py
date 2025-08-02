# E:\Diet Chatbot\rag\knowledge_base.py
import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

class KnowledgeBase:
    def __init__(self, embedding_model_name: str, google_api_key: str, vector_db_path: str = "./vector_db"):
        self.vector_db_path = vector_db_path
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=google_api_key)
        self.vectorstore = self._get_or_create_vectorstore()

    def _get_or_create_vectorstore(self):
        if os.path.exists(self.vector_db_path) and len(os.listdir(self.vector_db_path)) > 0:
            print(f"Loading existing vector store from {self.vector_db_path}")
            return Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
        else:
            print(f"Creating new vector store at {self.vector_db_path}")
            docs = []

            base_pdf_dir = "./data/recipe_pdfs"
            dietary_types = ["vegetarian", "vegan", "non_vegetarian"]

            for diet_type in dietary_types:
                pdf_dir = os.path.join(base_pdf_dir, diet_type)
                if os.path.exists(pdf_dir):
                    for filename in os.listdir(pdf_dir):
                        if filename.endswith(".pdf"):
                            pdf_path = os.path.join(pdf_dir, filename)
                            print(f"Loading PDF: {pdf_path} (Type: {diet_type})")
                            try:
                                loader = PyPDFLoader(pdf_path)
                                pdf_docs = loader.load()
                                for doc in pdf_docs:
                                    doc.metadata["source_file"] = filename
                                    doc.metadata["dietary_type"] = diet_type
                                    doc.metadata["doc_type"] = "recipe_book_pdf"
                                docs.extend(pdf_docs)
                            except Exception as e:
                                print(f"Error loading PDF {filename}: {e}")
                else:
                    print(f"Warning: {pdf_dir} not found. Skipping PDF loading for {diet_type}.")

            if not docs:
                print("No documents loaded for the vector store. Ensure your data directories exist and contain files.")
                return Chroma(embedding_function=self.embeddings, persist_directory=self.vector_db_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            print(f"Adding {len(splits)} chunks to the vector store.")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.vector_db_path
            )
            return vectorstore

    def get_retriever(self):
        return self.vectorstore.as_retriever()