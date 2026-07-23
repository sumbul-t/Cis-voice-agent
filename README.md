# CIS Department Voice Assistant

A voice-based agentic AI assistant I built for my Generative AI & Agentic AI course. You can talk to it (or type) and it answers questions about a university CIS department, quizzes you on a topic, summarizes text, translates, or calls a few utility tools — all through one conversational interface.

**Live demo:** (https://cis-voice-agent.onrender.com/)
*(hosted on Render's free tier — if it's been idle it takes ~30-60 seconds to wake up on the first request)*

## How it works

You ask a question by voice or text. The audio gets transcribed with OpenAI's Whisper, and then a router decides how to handle it:

- General questions about the department (courses, admissions, faculty, etc.) go through a **RAG pipeline** — the question gets embedded and matched against a knowledge base stored in **Pinecone**, and the retrieved context is passed into a **LangGraph** chatbot node (with memory across turns) that calls GPT-4o-mini to generate an answer.
- Utility requests like "what time is it," "convert 75F to Celsius," "look up CIS 360," or "calculate my GPA" get routed to tools on a separate **MCP server**.
- Task-style requests like "quiz me on binary search trees," "summarize this," or "translate this to Spanish" get routed to skill agents on an **A2A server**.

Whatever the answer is, it's spoken back out loud using OpenAI's TTS.

```
 mic / text input
        |
     Whisper (speech-to-text)
        |
    intent router
     /     |      \
  RAG    MCP      A2A
(Pinecone +      (tools:      (agents:
 LangGraph +      time, GPA,   quiz, summarize,
 GPT-4o-mini)     course...)   translate)
     \     |      /
        OpenAI TTS (speech out)
```

The MCP and A2A servers are separate services I deployed on Railway — this app just talks to them over HTTP.

## Stack

Python, Gradio, OpenAI (Whisper, GPT-4o-mini, TTS, embeddings), Pinecone, LangGraph, LangChain, MCP, A2A.

## Files

```
app.py              - Gradio app (entry point)
rag.py               - Whisper, TTS, embeddings, Pinecone retrieval, LangGraph graph
agents.py             - MCP/A2A calls + intent router
knowledge_base.py      - department knowledge base (18 chunks)
upload_kb.py             - one-time script to embed & upload the KB to Pinecone
config.py                  - loads API keys from environment variables
requirements.txt
```

## Running it locally

```bash
git clone https://github.com/sumbul-t/Cis-voice-agent.git
cd Cis-voice-agent
pip install -r requirements.txt

cp .env.example .env
# add your OPENAI_API_KEY and PINECONE_API_KEY to .env

# one-time: create a Pinecone index (1024 dims, cosine metric)
# named cis-department-kb, then run:
python upload_kb.py

python app.py
```

Then open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

## Deployment

Deployed on Render as a free web service, with `OPENAI_API_KEY` and `PINECONE_API_KEY` set as environment variables in Render's dashboard rather than in the code. Any push to `main` triggers an automatic redeploy.

## Notes

This started as a Colab notebook for a class project (`Project_1_to_Project_2.ipynb`, kept in the repo for reference) that used a browser-JS mic widget only Colab supports. I rebuilt the frontend in Gradio so it runs anywhere and can be deployed publicly.
