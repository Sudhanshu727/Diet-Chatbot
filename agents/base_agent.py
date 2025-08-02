# E:\Diet Chatbot\agents\base_agent.py
# REMOVE: import os, sys, sys.path.append(...)
# REMOVE: from app_config import GOOGLE_API_KEY, GEMINI_MODEL, TEMPERATURE

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import BaseMessage
from typing import List
# E:\Diet Chatbot\agents\base_agent.py

# ... (existing imports) ...
from langchain.agents import create_tool_calling_agent # Ensure this is present
from langchain.agents import AgentExecutor # Keep this if you want AgentExecutor for now, but we'll modify usage

class BaseDietAgent:
    def __init__(self, name: str, system_message: str, tools: List,
                 google_api_key: str, gemini_model: str, temperature: float):
        self.name = name
        self.llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=google_api_key,
            temperature=temperature
        )
        self.tools = tools
        self.system_message = system_message
        # Store the actual runnable agent here
        self.agent = self._create_runnable_agent() # <--- CHANGE THIS LINE
        # We might not even need agent_executor if we invoke self.agent directly
        # For simplicity, let's keep a wrapper if you prefer the AgentExecutor interface for now
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True) # <--- Keep this if needed elsewhere

    def _create_runnable_agent(self): # <--- CHANGE METHOD NAME
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_message),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        # This returns a Runnable
        return create_tool_calling_agent(self.llm, self.tools, prompt)

    def run(self, state):
        # This method uses agent_executor, which is fine if it works.
        # If the problem persists, we might switch this to self.agent.invoke directly.
        input_message = state["messages"][-1].content
        chat_history = state["messages"][:-1]
        result = self.agent_executor.invoke({"input": input_message, "chat_history": chat_history})
        return result["output"]

