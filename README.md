# 🗂️ Acme Policy Copilot — Enterprise HR & IT Policy Assistant

Acme Policy Copilot is a **Retrieval-Augmented Generation (RAG)** and **agentic tool-calling**
assistant that answers employee questions from a company's HR and IT policy documents, checks an
employee's leave balance, and files IT tickets — all orchestrated through a **LangGraph** workflow
using **LangChain** and a local **Chroma** vector database.

The application ingests policy documents into a local vector index, retrieves the relevant
sections for a given question, and lets an LLM (running via **Ollama** locally, or **Groq** as a
hosted alternative) decide — through native tool calling — whether to search the policy corpus,
look up an employee record, run a calculation, or file a ticket, before producing a grounded final
answer.

---

## ✨ Features

- 🗂️ RAG lookup over HR and IT policy documents using a local Chroma vector database
- 🧩 LangChain-based ingestion pipeline (loaders, chunking, embeddings, retriever)
- 🕸️ LangGraph orchestration with explicit branching, tool execution, and a step-limit safety net
- 🛠️ Native, schema-validated tool calling (no manual regex parsing)
- 👤 Mock HR leave-balance lookup tool
- 🎫 Mock IT ticket-creation tool
- 🧮 Safe AST-based calculator (no unsafe `eval()`)
- 🦙 Runs a real Llama model locally via Ollama — free forever, no API key, no rate limits
- ⚡ Optional Groq free-tier fallback provider
- 💬 Interactive Streamlit interface with a live tool-call reasoning trace
- 📊 Evaluation suite covering faithfulness, tool-selection accuracy, latency, and token usage
- 💰 Zero cost end-to-end — no paid API, no cloud account, no credit card anywhere

---

## 📸 Demo

The **`docs/`** folder is reserved for demonstration assets for this project, following the same
pattern as this author's other repositories:

- `demo.mov` – Walkthrough of the Streamlit app answering a policy question, checking a leave
  balance, and filing a ticket.
- `evaluation.png` – Screenshot of the evaluation suite's faithfulness and tool-selection results.

Add your own recording/screenshots to `docs/` after running the app locally.

---

## 🏗️ Architecture

```text
                    data/*.md (HR + IT policy documents)
                                 │
                                 ▼
                     ingest/build_index.py
        load → chunk (RecursiveCharacterTextSplitter)
        → embed (local HuggingFace MiniLM) → Chroma index
                                 │
                                 ▼
User question ──────────► LangGraph Agent
                                 │
                    ┌────────────┴────────────┐
                    │          agent            │◄──────┐
                    │  (Llama via Ollama/Groq,   │       │
                    │   native tool calling)      │       │
                    └────────────┬────────────┘       │
                       tool_calls present?               │
                    ┌────────────┴────────────┐         │
                   YES                          NO        │
                    │                            │         │
                    ▼                            ▼         │
            ┌───────────────┐                   END        │
            │     tools       │                             │
            │ search_policy_docs (RAG)                      │
            │ check_leave_balance (mock HR)                  │
            │ create_it_ticket (mock ITSM)                     │
            │ calculate (safe AST arithmetic)                   │
            └───────┬───────┘                                 │
                    └─────────────────────────────────────────┘
                                 │
                                 ▼
                    Streamlit Dashboard
          (reasoning trace + grounded final answer)
```

---

## 📂 Project Structure

```text
policy_copilot/
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
│
├── data/
│   ├── hr_policy.md
│   └── it_policy.md
│
├── ingest/
│   └── build_index.py
│
├── rag/
│   └── retriever.py
│
├── agent/
│   ├── llm.py
│   ├── tools.py
│   └── graph.py
│
└── eval/
    ├── faithfulness.py
    ├── run_eval.py
    └── test_tasks.json
```

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- Ollama (local Llama 3.1 8B)
- Groq API (optional hosted fallback)
- Chroma (local vector database)
- HuggingFace `sentence-transformers` (local embeddings)
- python-dotenv
- Python AST

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Moulya-Reddy/Acme-Policy-Copilot.git

cd Acme-Policy-Copilot
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```cmd
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up the LLM provider

**Recommended — Ollama (real Llama, local, free forever, no API key):**

```bash
brew install ollama
ollama pull llama3.1:8b
ollama serve
```

**Alternative — Groq (hosted, free tier):**

Create a `.env` file in the project root:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
```

### 6. Build the local policy index

```bash
python ingest/build_index.py
```

### 7. Run the application

```bash
streamlit run app.py
```

Open your browser and visit

```
http://localhost:8501
```

---

## 📊 Evaluation

Run the evaluation suite using

```bash
python eval/run_eval.py
```

The evaluation reports:

- Faithfulness of generated answers against retrieved policy text
- Tool selection accuracy
- Latency
- Token usage
- API cost

---

## 🎯 Design Decisions

### Why RAG?

Policy documents change over time and an LLM's training data can't reflect a company's actual,
current rules. Retrieving the relevant policy section before generation keeps answers grounded in
real, current documents instead of the model's memorized (and potentially outdated) knowledge.

### Why LangGraph instead of a simple linear chain?

The agent has genuine branching (RAG lookup vs. leave-balance lookup vs. ticket creation vs. no
tool at all) and a real loop (a tool result can lead to another tool call before a final answer).
LangGraph models this as an explicit, inspectable graph with a conditional exit and a step-limit
safety net, rather than an implicit procedural loop.

### Why native tool calling?

The model returns a structured, schema-validated tool call instead of free text that has to be
parsed — arguments like the IT ticket's severity level are constrained to a fixed set of valid
values at the schema level, which a plain-text approach cannot guarantee.

### Why Ollama as the default provider?

Groq deprecated every general-purpose Llama chat model it hosted in 2026. Running Llama locally
via Ollama avoids depending on any single hosted provider's model lifecycle, keeps the project
free forever with no API key, and is a genuine Llama model rather than a substitute.

### Why an AST-Based Calculator?

Using Python's `eval()` on any string that could ultimately be influenced by model or user input
is a code-execution risk. This project uses a restricted AST parser that only evaluates a strict
allow-list of arithmetic node types.

### Why Mock HR/IT Systems?

`check_leave_balance` and `create_it_ticket` stand in for what would be real, authenticated internal
API calls (an HRIS system, a ticketing system like ServiceNow) in production — kept obviously mock
so this project is safe to run and demo without connecting to any real enterprise system.

---

## 🔮 Future Improvements

- Real, authenticated HR/ITSM API integrations in place of the mock tools
- Human-in-the-loop approval before consequential tool actions
- Access-control-aware retrieval for multi-user, multi-department deployments
- Hybrid search (keyword + vector) and reranking as the policy corpus grows
- Persistent, cross-session conversation memory
- Multi-agent routing for distinct HR vs. IT specialist roles
- Authentication and per-user rate limiting on the Streamlit app
- Docker deployment

---

## 💡 Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- LangChain
- LangGraph
- Large Language Models (LLMs)
- Native Tool / Function Calling
- Agentic AI
- Prompt Engineering
- AI Evaluation
- Vector Databases
- API Integration
- Python Development
- Software Architecture
- Streamlit

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Moulya Reddy Kandhala**

GitHub: https://github.com/Moulya-Reddy
