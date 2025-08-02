# E:\Diet Chatbot\agents\orchestrator.py

from .base_agent import BaseDietAgent # CORRECT: Relative import
from .common_tools import tavily_search, retrieve_from_knowledge_base # CORRECT: Relative import

# Updated Pydantic import (ensure Pydantic v2 is installed, or use from pydantic.v1)
from pydantic import BaseModel, Field # <--- UPDATED IMPORT for Pydantic

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ADD THIS IMPORT: create_tool_calling_agent
from langchain.agents import create_tool_calling_agent # <--- ADDED THIS LINE
from langchain_core.output_parsers import JsonOutputParser # Keep this if it's used elsewhere

# Define a Pydantic model for structured output from the orchestrator
# E:\Diet Chatbot\agents\orchestrator.py

# ... (other imports) ...

from pydantic import BaseModel, Field
# Use Optional for fields that can be None (i.e., null from JSON)
from typing import Optional # <--- ADD THIS IMPORT

# Define a Pydantic model for structured output from the orchestrator
class RouteDecision(BaseModel):
    next_agent: str = Field(description="The name of the next agent to route to (e.g., 'vegetarian', 'non_vegetarian', 'vegan', 'general').")
    # Make these fields Optional[str]
    dietary_preference: Optional[str] = Field(default=None, description="Extracted dietary preference from the user's query (e.g., 'vegetarian', 'vegan', 'non_vegetarian', 'keto').")
    dietary_goal: Optional[str] = Field(default=None, description="Extracted dietary goal (e.g., 'weight loss', 'muscle gain', 'general health').")
    allergies: Optional[str] = Field(default=None, description="Comma-separated list of allergies (e.g., 'gluten,dairy').")
    meal_type: Optional[str] = Field(default=None, description="Type of meal requested (e.g., 'breakfast', 'lunch', 'dinner', 'snack').")
    query_for_agent: str = Field(description="The refined query to pass to the next agent.")

# ... (rest of OrchestratorAgent class) ...

class OrchestratorAgent(BaseDietAgent):
    def __init__(self, google_api_key: str, gemini_model: str, temperature: float):
        system_message = """You are the central routing agent for a diet suggestion chatbot.
        Your main task is to analyze the user's query and determine the most appropriate specialized diet agent (e.g., 'vegetarian', 'non_vegetarian', 'vegan') or
        if the request is general enough to be handled by general tools (like Tavily search for general facts).

        **IMPORTANT: If the user asks for a diet suggestion but does NOT specify their dietary preference (e.g., 'I want a dinner recipe'), you MUST ask them to clarify first (e.g., 'Are you looking for a vegetarian, vegan, or non-vegetarian recipe?').**

        You must extract key information like:
        - Dietary preference (e.g., vegetarian, vegan, non_vegetarian, keto, diabetic)
        - Dietary goal (e.g., weight loss, muscle gain, general health)
        - Any allergies or restrictions (e.g., gluten-free, dairy-free)
        - Type of meal (e.g., breakfast, lunch, dinner, snack)

        Based on this analysis, you will output a JSON object indicating:
        1. The `next_agent` to route to ('vegetarian', 'non_vegetarian', 'vegan', 'general').
           - Use 'vegetarian' if the user explicitly states they are vegetarian or implies it.
           - Use 'vegan' if the user explicitly states they are vegan.
           - Use 'non_vegetarian' if the user explicitly states they are non-vegetarian or implies it.
           - Use 'general' if the query is for general diet facts, or if no specific dietary preference is mentioned for a diet suggestion AND you have asked for clarification.
        2. The `dietary_preference` extracted.
        3. The `dietary_goal` extracted.
        4. The `allergies` extracted.
        5. The `meal_type` extracted.
        6. A `query_for_agent` which is a concise restatement of the user's core request to pass to the next agent.

        Always provide a `query_for_agent`.
        If you need to ask for clarification, ensure your response is a question to the user and the `next_agent` is 'general' for that turn, then the user's follow-up will re-enter the orchestrator.

        Example JSON output:
        {{
            "next_agent": "vegetarian",
            "dietary_preference": "vegetarian",
            "dietary_goal": "weight loss",
            "allergies": "none",
            "meal_type": "dinner",
            "query_for_agent": "vegetarian dinner ideas for weight loss"
        }}
        """
        tools = [tavily_search, retrieve_from_knowledge_base]

        super().__init__(
            name="Orchestrator",
            system_message=system_message,
            tools=tools,
            google_api_key=google_api_key,
            gemini_model=gemini_model,
            temperature=temperature
        )
        self.parser = JsonOutputParser(pydantic_object=RouteDecision)



    def parse_decision(self, raw_output: str) -> RouteDecision:
        try:
            return self.parser.parse(raw_output)
        except Exception as e:
            print(f"Orchestrator output not JSON (likely a clarifying question/direct response): {e}\nRaw output: {raw_output}")
            return RouteDecision(
                next_agent="general",
                dietary_preference="",
                dietary_goal="",
                allergies="",
                meal_type="",
                query_for_agent=raw_output
            )