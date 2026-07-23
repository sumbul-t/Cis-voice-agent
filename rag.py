"""
Core RAG pipeline: OpenAI embeddings + Pinecone retrieval + LangGraph
conversational LLM node. Adapted from the original research notebook
into importable, testable functions.
"""
from openai import OpenAI
from pinecone import Pinecone
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

RAG_PROMPT = """\
You are a helpful voice assistant for the Computer and Information Science (CIS) department
at UMass Dartmouth. Answer the student's question using ONLY the context provided below.
If the context does not contain enough information to answer fully, say so honestly
and suggest the student contact the department directly.

Keep your answers clear, friendly, and concise — they will be read aloud as speech.
Avoid bullet points or markdown formatting; use plain spoken sentences.

CONTEXT:
{context}
"""


def embed_text(text: str) -> list[float]:
    """Embed a string into a 1024-dim vector matching the Pinecone index."""
    response = client.embeddings.create(
        model="text-embedding-3-small", input=text, dimensions=1024
    )
    return response.data[0].embedding


def retrieve_context(question: str, top_k: int = 3) -> str:
    """Embed the question and pull the top_k most relevant chunks from Pinecone."""
    question_vector = embed_text(question)
    results = index.query(
        vector=question_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
    )
    context_pieces = [match["metadata"]["text"] for match in results["matches"]]
    return "\n\n".join(context_pieces)


def transcribe(audio_path: str) -> str:
    """Speech-to-text via OpenAI Whisper."""
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="text"
        )
    return response.strip()


def speak(text: str, voice: str = "nova", output_path: str = "response.mp3") -> str:
    """Text-to-speech via OpenAI TTS. Returns the saved audio file path."""
    response = client.audio.speech.create(model="tts-1", voice=voice, input=text)
    response.stream_to_file(output_path)
    return output_path


def _rag_node(state: MessagesState) -> dict:
    user_message = state["messages"][-1].content
    context = retrieve_context(user_message, top_k=3)
    system_prompt = RAG_PROMPT.format(context=context)

    messages_for_llm = [{"role": "system", "content": system_prompt}]
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            messages_for_llm.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages_for_llm.append({"role": "assistant", "content": msg.content})

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages_for_llm
    )
    answer = response.choices[0].message.content
    return {"messages": state["messages"] + [AIMessage(content=answer)]}


_memory = MemorySaver()
_graph_builder = StateGraph(MessagesState)
_graph_builder.add_node("rag_chatbot", _rag_node)
_graph_builder.set_entry_point("rag_chatbot")
_graph_builder.add_edge("rag_chatbot", END)
chatbot_graph = _graph_builder.compile(checkpointer=_memory)


def ask_rag(question: str, thread_id: str = "default-session") -> str:
    """Run one turn through the LangGraph RAG chatbot, keeping per-thread memory."""
    result = chatbot_graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content
