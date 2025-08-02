# E:\Diet Chatbot\agents\vegetarian.py
# ... (your existing imports) ...
from .base_agent import BaseDietAgent
from .common_tools import retrieve_from_knowledge_base, tavily_search

class VegetarianDietAgent(BaseDietAgent):
    # Add the config parameters to the __init__ signature
    def __init__(self, google_api_key: str, gemini_model: str, temperature: float):
        system_message = """You are an expert vegetarian diet planning assistant.
        Provide healthy and delicious vegetarian meal ideas, recipes, and dietary advice.
        Focus on plant-based protein sources, balanced nutrition, and user preferences.
        Use your tools to find relevant information from the knowledge base or web search.
        """
        tools = [retrieve_from_knowledge_base, tavily_search] # Vegetarian agent tools

        # Pass all required parameters to the BaseDietAgent's constructor
        super().__init__(
            name="Vegetarian Diet Agent",
            system_message=system_message,
            tools=tools,
            google_api_key=google_api_key, # Pass this
            gemini_model=gemini_model,     # Pass this
            temperature=temperature        # Pass this
        )