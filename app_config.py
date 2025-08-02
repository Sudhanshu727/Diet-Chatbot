import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LLM Configuration
GEMINI_MODEL = "gemini-1.5-flash" 
TEMPERATURE = 0.9 

# RAG Configuration
VECTOR_DB_PATH = "./vector_db" # Path to store ChromaDB persistent collection