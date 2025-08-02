# E:\Diet Chatbot\main.py
import json
import operator
from typing import List, Tuple, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import from your top-level app_config
from app_config import GOOGLE_API_KEY, GEMINI_MODEL, TEMPERATURE, VECTOR_DB_PATH, TAVILY_API_KEY

# Import agents (absolute paths from the package root, assuming 'Diet Chatbot' is the root)
from agents.orchestrator import OrchestratorAgent, RouteDecision
from agents.vegetarian import VegetarianDietAgent
from agents.non_vegetarian import NonVegetarianDietAgent
from agents.vegan import VeganDietAgent
from agents.base_agent import BaseDietAgent # Used for the general agent
from agents.common_tools import tavily_search, retrieve_from_knowledge_base

# Import RAG components
from rag.knowledge_base import KnowledgeBase # <--- ADDED: Need to import KnowledgeBase here to initialize it
from rag.retriever import set_rag_retriever # Only need set_rag_retriever here

from langgraph.graph import StateGraph, END
import os

# --- Initialize RAG Knowledge Base and Retriever ---
# E:\Diet Chatbot\main.py

# ... (other imports) ...

# Import common_tools setters
from agents.common_tools import set_global_rag_retriever, set_global_tavily_tool # <--- ADD THIS IMPORT

# Import TavilySearchResults here to initialize it
from langchain_community.tools.tavily_search import TavilySearchResults # <--- ADD THIS IMPORT
from app_config import GOOGLE_API_KEY, GEMINI_MODEL, TEMPERATURE, VECTOR_DB_PATH, TAVILY_API_KEY


# --- Initialize RAG Knowledge Base and Retriever ---
print("Initializing knowledge base (this may take a while the first time)...")
knowledge_base_instance = KnowledgeBase(
    embedding_model_name="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
    vector_db_path=VECTOR_DB_PATH
)
# Set the RAG retriever for common_tools
set_global_rag_retriever(knowledge_base_instance.get_retriever()) # <--- ADD THIS LINE
print("Knowledge base ready!")

# --- Initialize Tavily Tool ---
if TAVILY_API_KEY:
    tavily_tool_instance = TavilySearchResults(api_key=TAVILY_API_KEY)
    set_global_tavily_tool(tavily_tool_instance) # <--- ADD THIS LINE
    print("Tavily search tool ready!")
else:
    print("TAVILY_API_KEY not set. Tavily search will not be available.")



# Define the state for LangGraph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    dietary_preference: str
    dietary_goal: str
    allergies: List[str]
    meal_type: str
    next_agent_route: str
    query_for_next_agent: str

# Initialize agents, PASSING CONFIG VARIABLES
orchestrator = OrchestratorAgent(
    google_api_key=GOOGLE_API_KEY, gemini_model=GEMINI_MODEL, temperature=TEMPERATURE
)
vegetarian_agent = VegetarianDietAgent(
    google_api_key=GOOGLE_API_KEY, gemini_model=GEMINI_MODEL, temperature=TEMPERATURE
)
non_vegetarian_agent = NonVegetarianDietAgent(
    google_api_key=GOOGLE_API_KEY, gemini_model=GEMINI_MODEL, temperature=TEMPERATURE
)
vegan_agent = VeganDietAgent(
    google_api_key=GOOGLE_API_KEY, gemini_model=GEMINI_MODEL, temperature=TEMPERATURE
)

# A "general" agent for fallback or general queries
general_agent = BaseDietAgent(
    name="General Diet Agent",
    system_message="You are a helpful diet assistant providing general information and advice. Use your tools to find answers.",
    tools=[tavily_search, retrieve_from_knowledge_base],
    google_api_key=GOOGLE_API_KEY, gemini_model=GEMINI_MODEL, temperature=TEMPERATURE # Pass configs here too
)

# E:\Diet Chatbot\main.py


# --- Define Nodes ---
def call_orchestrator(state: AgentState):
    print("\n--- Calling Orchestrator ---")
    user_message = state["messages"][-1].content
    orchestrator_input_state = {"input": user_message, "chat_history": state["messages"][:-1]}

    orchestrator_result = orchestrator.agent_executor.invoke(orchestrator_input_state)
    raw_llm_output_content = orchestrator_result['output'] # This can be a dict, string with JSON, or plain string

    decision = None
    parsed_successfully = False

    # First, try to parse it directly as JSON if it's a string, removing markdown wrappers
    if isinstance(raw_llm_output_content, str):
        json_string = raw_llm_output_content.strip()
        # Remove markdown code blocks if present
        if json_string.startswith("```json"):
            json_string = json_string[len("```json"):].strip()
        if json_string.endswith("```"):
            json_string = json_string[:-len("```")].strip()

        try:
            # Try to load as JSON
            raw_output_dict = json.loads(json_string)
            decision = RouteDecision(**raw_output_dict)
            parsed_successfully = True
            print(f"Orchestrator parsed decision (from string JSON): {decision}")
        except json.JSONDecodeError:
            # If it's a string but not valid JSON after stripping, treat as plain text
            print(f"Orchestrator returned plain string (not valid JSON): {raw_llm_output_content}")
            decision = RouteDecision(next_agent="general", query_for_agent=raw_llm_output_content)
            parsed_successfully = True # Successfully handled as plain string
        except Exception as e: # Catch Pydantic errors if they still happen here
            print(f"Orchestrator string output could not be parsed into RouteDecision: {e}. Output was: {raw_llm_output_content}")
            # Fallback to general agent with the raw output if Pydantic parsing fails
            decision = RouteDecision(next_agent="general", query_for_agent=raw_llm_output_content)
            parsed_successfully = True

    elif isinstance(raw_llm_output_content, dict):
        # If the output is already a dictionary, try to parse it into RouteDecision
        try:
            decision = RouteDecision(**raw_llm_output_content)
            parsed_successfully = True
            print(f"Orchestrator parsed decision (from dict): {decision}")
        except Exception as e:
            print(f"Orchestrator output dict could not be parsed into RouteDecision: {e}. Output was: {raw_llm_output_content}")
            # Fallback to general agent if Pydantic parsing fails from dict
            decision = RouteDecision(next_agent="general", query_for_agent=str(raw_llm_output_content))
            parsed_successfully = True

    if not parsed_successfully or decision is None:
        # Final fallback if none of the above worked
        print(f"Orchestrator returned unexpected type or failed final parsing: {type(raw_llm_output_content)}. Output: {raw_llm_output_content}")
        decision = RouteDecision(next_agent="general", query_for_agent="I'm sorry, I encountered an unexpected routing error.")

    # --- Refined Routing Logic ---
    # Only route to "general" if the decision explicitly says "general"
    # AND the query_for_agent is clearly a greeting/clarification.
    # Otherwise, trust the next_agent from the LLM's JSON.
    if decision.next_agent == "general" and (
        "help you with today" in decision.query_for_agent.lower() or
        "what kind of diet information" in decision.query_for_agent.lower() or
        "how can i assist you" in decision.query_for_agent.lower()
    ):
        display_message = decision.query_for_agent
        # If, for some reason, the query_for_agent still contains a JSON string,
        # ensure we present a friendly, generic message.
        if "next_agent" in display_message and "query_for_agent" in display_message:
             display_message = "What can I help you with today?" # Default friendly greeting
        print(f"Orchestrator issued a general greeting/clarification: {display_message}")
        return {
            "messages": [AIMessage(content=display_message)],
            "next_agent_route": "general",
            "query_for_next_agent": display_message
        }
    else:
        # If the orchestrator provided a specific agent (e.g., "vegan", "non_vegetarian")
        # or a specific, non-greeting query for the general agent, route accordingly.
        print(f"Orchestrator Decision: {decision}")
        state["dietary_preference"] = decision.dietary_preference
        state["dietary_goal"] = decision.dietary_goal
        state["allergies"] = [a.strip() for a in decision.allergies.split(',')] if decision.allergies else []
        state["meal_type"] = decision.meal_type

        # The message content should be user-friendly, not the raw routing decision.
        # It's good practice to provide feedback to the user about routing.
        response_message = f"Routing you to the {decision.next_agent} agent for '{decision.query_for_agent}'."
        if decision.next_agent == "general":
            # If it's general but not a greeting, just use the query itself
            response_message = f"Processing your general query: '{decision.query_for_agent}'."

        return {
            "messages": [AIMessage(content=response_message)],
            "next_agent_route": decision.next_agent,
            "query_for_next_agent": decision.query_for_agent
        }



def call_vegetarian_agent(state: AgentState):
    print("\n--- Calling Vegetarian Agent ---")
    query = state.get("query_for_next_agent", state["messages"][-1].content)
    agent_input_state = {
        "input": query,
        "chat_history": state["messages"]
    }
    response = vegetarian_agent.agent_executor.invoke(agent_input_state)["output"]
    return {"messages": [AIMessage(content=response)]}

def call_non_vegetarian_agent(state: AgentState):
    print("\n--- Calling Non-Vegetarian Agent ---")
    query = state.get("query_for_next_agent", state["messages"][-1].content)
    agent_input_state = {
        "input": query,
        "chat_history": state["messages"]
    }
    response = non_vegetarian_agent.agent_executor.invoke(agent_input_state)["output"]
    return {"messages": [AIMessage(content=response)]}

def call_vegan_agent(state: AgentState):
    print("\n--- Calling Vegan Agent ---")
    query = state.get("query_for_next_agent", state["messages"][-1].content)
    agent_input_state = {
        "input": query,
        "chat_history": state["messages"]
    }
    response = vegan_agent.agent_executor.invoke(agent_input_state)["output"]
    return {"messages": [AIMessage(content=response)]}

def call_general_agent(state: AgentState):
    print("\n--- Calling General Agent ---")
    query = state.get("query_for_next_agent", state["messages"][-1].content)
    agent_input_state = {
        "input": query,
        "chat_history": state["messages"]
    }
    response = general_agent.agent_executor.invoke(agent_input_state)["output"]
    return {"messages": [AIMessage(content=response)]}


# --- Define Router ---
def route_agent(state: AgentState):
    next_agent_route = state.get("next_agent_route")
    if next_agent_route:
        print(f"Routing to: {next_agent_route}")
        return next_agent_route
    else:
        print("No routing decision made by orchestrator, falling back to general.")
        return "general"

# --- Build the LangGraph ---
workflow = StateGraph(AgentState)

workflow.add_node("orchestrator", call_orchestrator)
workflow.add_node("vegetarian", call_vegetarian_agent)
workflow.add_node("non_vegetarian", call_non_vegetarian_agent)
workflow.add_node("vegan", call_vegan_agent)
workflow.add_node("general", call_general_agent)

workflow.set_entry_point("orchestrator")

workflow.add_conditional_edges(
    "orchestrator",
    route_agent,
    {
        "vegetarian": "vegetarian",
        "non_vegetarian": "non_vegetarian",
        "vegan": "vegan",
        "general": "general"
    }
)

workflow.add_edge("vegetarian", END)
workflow.add_edge("non_vegetarian", END)
workflow.add_edge("vegan", END)
workflow.add_edge("general", END)

app = workflow.compile()

# --- Example Usage ---
if __name__ == "__main__":
    print("Diet Chatbot started. Type 'exit' to quit.")

    # Removed the get_rag_retriever call here, as it's now initialized above.
    # The get_rag_retriever call in common_tools.py will now correctly fetch the initialized retriever.

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        initial_state = {"messages": [HumanMessage(content=user_input)],
                         "dietary_preference": "",
                         "dietary_goal": "",
                         "allergies": [],
                         "meal_type": "",
                         "next_agent_route": "",
                         "query_for_next_agent": ""}
        try:
            final_state = app.invoke(initial_state)
            ai_response = final_state["messages"][-1].content
            print(f"Bot: {ai_response}")
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            print("Bot: I'm sorry, I couldn't process that request right now. Please try again or rephrase.")