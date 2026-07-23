# 🎓 CIS Department Voice Assistant — Agentic RAG System

A voice-driven agentic AI assistant that answers questions about a university
CIS department, and can also quiz you, summarize text, translate, and call
utility tools — all through a single conversational interface.

**[🔗 Live demo](#)** — *add your deployed Hugging Face Space link here*

## What it does

1. **Voice or text input** — record a question in the browser, or type it.
2. **Speech-to-text** via OpenAI Whisper.
3. **Intent routing** decides how to answer:
   - Open-ended department questions → **RAG pipeline**: OpenAI embeddings +
     **Pinecone** vector search retrieve relevant knowledge-base chunks,
     which are injected into a **LangGraph** stateful chatbot node (with
     per-session memory) that calls GPT-4o-mini for a grounded answer.
   - Utility requests ("what time is it", "convert 75F to Celsius", "look up
     CIS 360", "calculate my GPA") → routed to tools on a separate **MCP
     (Model Context Protocol) server**, with response caching.
   - Task requests ("quiz me on binary search trees", "summarize this",
     "translate to Spanish") → routed to skill-specific agents on an
     **A2A (agent-to-agent) server**.
4. **Text-to-speech** via OpenAI TTS turns the answer into a spoken reply.

## Architecture

```
                     ┌───────────────────────┐
   🎙️ / ⌨️  input ──▶ │   Whisper (speech-to-  │
                     │        text)          │
                     └──────────┬────────────┘
                                ▼
                     ┌───────────────────────┐
                     │     Intent Router      │
                     └──┬──────────┬───────┬─┘
             ┌──────────┘          │       └──────────┐
             ▼                     ▼                  ▼
   ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐
   │  Pinecone + RAG   │  │   MCP tools      │  │   A2A agents     │
   │  (LangGraph +     │  │  (time, GPA,     │  │  (quiz, summary, │
   │   GPT-4o-mini)    │  │  course lookup…) │  │   translate)      │
   └──────────────────┘  └─────────────────┘  └──────────────────┘
             │                     │                  │
             └──────────┬──────────┴──────────────────┘
                         ▼
              ┌───────────────────────┐
              │   OpenAI TTS (voice)   │
              └───────────────────────┘
```

The MCP and A2A servers are separate deployed services (this repo just
calls them over HTTP) — see `MCP_SERVER_URL` / `A2A_SERVER_URL` in
`config.py`.

## Tech stack

`Python` · `Gradio` · `OpenAI (Whisper, GPT-4o-mini, TTS, Embeddings)` ·
`Pinecone` · `LangGraph` · `LangChain` · `MCP` · `A2A`

## Project structure

```
├── app.py            # Gradio web app (entry point)
├── rag.py             # Whisper, TTS, embeddings, Pinecone retrieval, LangGraph graph
├── agents.py           # MCP tool calls, A2A agent calls, intent router
├── knowledge_base.py    # Department knowledge base (18 text chunks)
├── upload_kb.py          # One-time script to embed & upload the KB to Pinecone
├── config.py               # Loads API keys/settings from environment variables
├── requirements.txt
└── .env.example
```

## Running locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your OPENAI_API_KEY and PINECONE_API_KEY

# one-time: create a Pinecone index (1024 dimensions, cosine metric)
# named cis-department-kb in the Pinecone console, then run:
python upload_kb.py

python app.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

## Deploying it live (free, public URL)

The easiest path is **Hugging Face Spaces** (free tier, gives you a real
public link like `https://huggingface.co/spaces/<you>/<space-name>`):

1. Create a new Space at huggingface.co/new-space, SDK = **Gradio**.
2. Push this repo's files to the Space (it's a git repo — `git remote add
   space https://huggingface.co/spaces/<you>/<space-name>` then `git push
   space main`), or upload the files via the web UI.
3. In the Space's **Settings → Repository secrets**, add `OPENAI_API_KEY`
   and `PINECONE_API_KEY`.
4. The Space will build and give you a live URL automatically.

Alternatives that work just as well: **Render** or **Railway** (Docker/Web
Service, same env vars), since `app.py` also runs fine as a plain Python
web process.

## Notes

- This started as a research notebook (`Project_1_to_Project_2.ipynb`,
  included for reference) built for a Google Colab environment with a
  browser-JS microphone widget. This repo replaces that with a portable
  Gradio UI so it runs anywhere and can be deployed publicly.
- API keys are never hardcoded — everything reads from environment
  variables so the repo is safe to make public.
