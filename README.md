# Diet Chatbot

A modular, retrieval-augmented chatbot for personalized diet and recipe recommendations. Supports vegetarian, vegan, and non-vegetarian diets, leveraging LLMs, web search, and a custom PDF-based knowledge base.

---

## Features

- **Multiple Diet Agents:** Specialized agents for vegetarian, vegan, and non-vegetarian diets.
- **Retrieval-Augmented Generation (RAG):** Answers are grounded in your own recipe PDFs and dietary facts.
- **Web Search Integration:** Uses Tavily for up-to-date information.
- **Extensible:** Easily add new agents, tools, or knowledge sources.

---

## Project Structure

```
diet_chatbot/
├── .env                          # Environment variables (API keys)
├── main.py                       # Orchestrates the LangGraph workflow
├── agents/                       # Agent definitions and shared tools
│   ├── base_agent.py
│   ├── orchestrator.py
│   ├── vegetarian.py
│   ├── non_vegetarian.py
│   ├── vegan.py
│   └── common_tools.py
├── data/
│   ├── recipe_pdfs/              # PDF recipe books (by diet type)
│   ├── recipes.json              # (Optional) Structured recipe data
│   └── dietary_facts.json        # (Optional) Dietary facts
├── rag/
│   ├── knowledge_base.py         # PDF loading, chunking, vector store
│   └── retriever.py              # RAG retriever instance
├── config.py                     # Global configurations
└── requirements.txt              # Python dependencies
```

---

## Setup

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/diet_chatbot.git
   cd diet_chatbot
   ```

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Prepare your [`.env`](.env) file:**

   - Copy `.env.example` to [`.env`](.env) and fill in your API keys (Google, Tavily, etc.).

4. **Add your recipe PDFs:**
   - Place your recipe books in the appropriate folders under [`data/recipe_pdfs`](data/recipe_pdfs).

---

## Usage

Run the main script:

```sh
python main.py
```

The chatbot will initialize the knowledge base and agents. Interact with the chatbot as prompted.

---

## Customization

- **Add new recipes:** Place additional PDFs in [`data/recipe_pdfs`](data/recipe_pdfs).
- **Add new agents/tools:** Extend the classes in [`agents`](agents) and [`rag`](rag).
- **Change LLM model or API keys:** Edit `config.py` or your [`.env`](.env) file.

---

## License

MIT License

---

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [Tavily Search](https://www.tavily.com/)
- [Google Generative AI](https://ai.google/discover/generative-ai/)

---

**Happy healthy eating!**
