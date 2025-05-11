# 📑 CONSORT-RCT Assistant – AI-Powered Chatbot for Clinical Trial Reporting

Welcome to **CONSORT-RCT Assistant**, a domain-specific AI assistant built to streamline clinical trial research workflows! This chatbot leverages **LangChain**, **LangGraph**, and **Chainlit** to answer CONSORT and RCT-related questions using a powerful **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 🚀 Features

- 🔎 **Semantic Search with Self-Query Retriever**  
  Retrieve relevant research papers using OpenAI embeddings and metadata filtering.

- 🧠 **Context-Aware Answering**  
  Generates answers using a history-aware LLM with hallucination checks.

- 📊 **SQL-Based Analytics**  
  Automatically generate and execute SQL queries over your PostgreSQL-backed metadata database.

- 🧪 **Grounding Score & Hallucination Guard**  
  Each answer is validated for factual consistency using LLM-based JSON scoring.

- 🔄 **LangGraph Agent**  
  Integrates RAG and SQL tools into a revisable multi-node flow with automatic error handling.

- 🌐 **Chainlit Frontend with OAuth & Password Auth**  
  Simple and secure interface for interacting with the assistant.

---

## 🛠️ Tech Stack

- **LangChain & LangGraph** – Tool orchestration and graph-based agent logic  
- **Chainlit** – Realtime chat interface and session management  
- **PostgreSQL + PGVector** – Document storage and retrieval  
- **OpenAI GPT-4.1-mini / GPT-3.5** – LLMs for reasoning, retrieval, and verification  
- **Docker (optional)** – Containerization for deployment  

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/consort-rct-assistant.git
cd consort-rct-assistant

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Environment Variables

Set the following variables in a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4.1-mini
PG_URI=postgresql://username:password@host/dbname
PG_VECTOR_CONN=postgresql+psycopg://username:password@host/dbname
PG_VECTOR_COLLECTION=state_of_union_vectors
```

---

## ▶️ Run the App

```bash
# Launch Chainlit app
chainlit run app.py --port 8000
```

Then open `http://localhost:8000` in your browser.

---

## 📁 Project Structure

```bash
.
├── app.py                # Main Chainlit app with RAG + LangGraph integration
├── agent_graph.py        # LangGraph agent with SQL & RAG tools
├── requirements.txt      # Python dependencies
├── .gitignore            # Git exclusion list
├── chainlit.md           # User-facing welcome message
└── README.md             # You're here!
```

---

## 🔐 Authentication

Supports:

- ✅ Username/password: `admin` / `admin`  
- ✅ OAuth (extendable)

---

## 🧠 Sample Query

> *"What studies in 2020 mention the Intervention component of PICOS?"*

- Retrieves relevant documents
- Grades source relevance
- Validates groundedness
- Provides clear answer + source list

---

## 🧬 Use Case

Ideal for:

- Researchers exploring **brain-heart interconnectome**, **clinical trial reporting**, or **PICOS compliance**.
- Institutions aiming to **automate evidence synthesis** or **monitor research trends**.

---

## 💡 Inspiration

This project supports the mission of improving scientific transparency, accuracy, and reproducibility using state-of-the-art AI.

---

## 📚 Related

- 🔗 [LangChain Documentation](https://docs.langchain.com)
- 🔗 [Chainlit Docs](https://docs.chainlit.io)
- 🔗 [LangGraph](https://github.com/langchain-ai/langgraph)

---

## 📄 License

MIT License © 2025

---

## 🙌 Acknowledgments

Developed by [Pouria Mortezaagha](https://www.linkedin.com/in/pouria-mortezaagha/), in collaboration with Ottawa Hospital Research Institute.