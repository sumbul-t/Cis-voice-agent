"""
CIS Department Voice Assistant — live web app.

Pipeline: Voice/Text In -> Whisper Transcription -> Intent Router
          -> (Pinecone RAG | MCP tool | A2A agent) -> OpenAI TTS -> Voice Out

Run locally:
    pip install -r requirements.txt
    cp .env.example .env   # then fill in your API keys
    python app.py

Deploy: push this repo to a Hugging Face Space (Gradio SDK) and add
OPENAI_API_KEY / PINECONE_API_KEY as Space secrets. See README.md.
"""
import sys
import uuid

print("[startup] app.py starting...", flush=True)

import gradio as gr
print("[startup] gradio imported", flush=True)

from rag import transcribe, speak
print("[startup] rag imported (openai + pinecone connected)", flush=True)

from agents import route
print("[startup] agents imported", flush=True)

# One session/thread id per browser tab, so LangGraph's MemorySaver keeps
# separate conversation memory per visitor instead of one shared thread.
SESSION_ID = str(uuid.uuid4())


def handle_turn(audio_path, typed_text, history):
    history = history or []

    if audio_path:
        question = transcribe(audio_path)
    elif typed_text and typed_text.strip():
        question = typed_text.strip()
    else:
        return history, None, "", None

    answer = route(question, thread_id=SESSION_ID)
    audio_out = speak(answer, voice="nova", output_path="response.mp3")

    history = history + [(question, answer)]
    return history, audio_out, "", None


with gr.Blocks(title="CIS Department Voice Assistant") as demo:
    gr.Markdown(
        """
        # 🎓 CIS Department Voice Assistant
        Ask about the UMass Dartmouth CIS program, courses, GPA, campus info,
        or say **"quiz me on binary search trees"**, **"summarize ..."**,
        **"translate ... to Spanish"**, **"what time is it"**, and more.

        Record a question with your mic, or type it below.
        """
    )

    chatbot = gr.Chatbot(label="Conversation", height=400)

    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Ask by voice")
        text_in = gr.Textbox(label="Or type your question", placeholder="e.g. What degrees does the CIS department offer?")

    audio_out = gr.Audio(label="🔊 Spoken answer", autoplay=True)
    submit_btn = gr.Button("Ask", variant="primary")

    submit_btn.click(
        handle_turn,
        inputs=[mic, text_in, chatbot],
        outputs=[chatbot, audio_out, text_in, mic],
    )
    text_in.submit(
        handle_turn,
        inputs=[mic, text_in, chatbot],
        outputs=[chatbot, audio_out, text_in, mic],
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    print(f"[startup] launching Gradio on 0.0.0.0:{port}...", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=port)
    print("[startup] demo.launch() returned (should not normally happen)", flush=True)
