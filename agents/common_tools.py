# E:\Diet Chatbot\agents\common_tools.py
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


# We will use the helper functions from rag.retriever to get the current retriever
from rag.retriever import get_rag_retriever

# --- Global holders for initialized tools ---
# These will be set by main.py
_tavily_search_tool = None
_rag_retriever_instance = None

def set_global_tavily_tool(tool_instance):
    global _tavily_search_tool
    _tavily_search_tool = tool_instance

def set_global_rag_retriever(retriever_instance):
    global _rag_retriever_instance
    _rag_retriever_instance = retriever_instance

# --- Shared Tools Definitions ---

@tool
def tavily_search(query: str) -> str:
    """Use this tool to perform a general web search for information.
    Useful for looking up current events, general facts, or things not in the internal knowledge base.
    """
    if _tavily_search_tool is None:
        raise RuntimeError("Tavily search tool not initialized. Call set_global_tavily_tool from main.py first.")
    print(f"Performing Tavily search for: {query}")
    return _tavily_search_tool.invoke({"query": query})

# E:\Diet Chatbot\agents\common_tools.py

# ... (existing imports) ...

@tool
def retrieve_from_knowledge_base(query: str, dietary_filter: str = "") -> str:
    """
    Retrieves relevant information from the diet knowledge base based on the query and an optional dietary filter.
    Useful for finding recipes, nutritional facts, diet plans, etc.
    The dietary_filter should be 'vegetarian', 'non_vegetarian', or 'vegan' if applicable.
    Example: retrieve_from_knowledge_base(query="chicken breast recipes", dietary_filter="non_vegetarian")
    Example: retrieve_from_knowledge_base(query="benefits of mediterranean diet")
    """
    global _rag_retriever_instance
    if _rag_retriever_instance is None:
        raise RuntimeError("RAG retriever not initialized. Call set_global_rag_retriever from main.py first.")

    where_clauses = {}
    # Always try to filter by 'doc_type' if you expect it from a specific document.
    # Assuming your loaded documents have this metadata field.
    # Ensure this matches the metadata you added when ingesting documents.
    where_clauses["doc_type"] = "recipe_book_pdf" # Assuming all recipes are from this source

    if dietary_filter:
        # For an exact match on a metadata field, Chroma expects: {"field_name": "value"}
        # For multiple conditions, you must use "$and" or "$or"
        where_clauses["dietary_type"] = dietary_filter # This will be added to the existing dict

    # Now, combine the conditions using "$and" if there's more than one.
    # ChromaDB's where clause expects a single key, which can be an operator like "$and"
    final_where_clause = {}
    if len(where_clauses) > 1: # If both doc_type and dietary_type are present
        final_where_clause["$and"] = []
        for key, value in where_clauses.items():
            final_where_clause["$and"].append({key: value})
    elif len(where_clauses) == 1: # If only one condition (e.g., just doc_type or just dietary_type)
        final_where_clause = where_clauses # Use the single condition directly
    else: # No filters
        final_where_clause = {} # No filters applied

    print(f"Retrieving from knowledge base for: '{query}' with filter: {final_where_clause}")

    # Pass the correctly formatted filter to the retriever
    # Note: `get_relevant_documents` is deprecated, but we'll stick to it for now
    # until the main issues are resolved.
    docs = _rag_retriever_instance.get_relevant_documents(query, **({'filter': final_where_clause} if final_where_clause else {}))

    if not docs:
        return "No relevant information found in the knowledge base."

    # Concatenate document content
    content = "\n\n".join([doc.page_content for doc in docs])
    return content