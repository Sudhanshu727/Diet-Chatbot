from .base_agent import BaseDietAgent
from .common_tools import retrieve_from_knowledge_base, tavily_search
class VeganDietAgent(BaseDietAgent):
    def __init__(self, google_api_key: str, gemini_model: str, temperature: float):
        system_message = """You are an expert in vegan nutrition and meal planning.
        Your goal is to provide delicious, balanced, and healthy meal suggestions or recipes that are strictly vegan (no meat, milk, poultry, fish, dairy, eggs, or honey).
        Ensure all suggestions are 100% plant-based.
        **Crucially, use the `retrieve_from_knowledge_base` tool with `dietary_filter='vegan'` to find relevant recipes and information specifically from the vegan recipe book.**
        Use `tavily_search` for general web inquiries not covered by your internal knowledge.
        Always consider the user's dietary goals and allergies if provided. Be polite and helpful.
        """
        tools = [retrieve_from_knowledge_base, tavily_search]
        super().__init__(
            name="Vegetarian Diet Agent",
            system_message=system_message,
            tools=tools,
            google_api_key=google_api_key, # Pass this
            gemini_model=gemini_model,     # Pass this
            temperature=temperature        # Pass this
        )