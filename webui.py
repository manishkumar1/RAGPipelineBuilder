"""
RAG Pipeline Web UI
Run with: python webui.py
"""
import os
import gradio as gr
from dotenv import load_dotenv

from data.constants import groq_api_key, vector_db_name, embedding_model_name
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_retriever import RAGRetriever
from src.rag_pipeline import RAGPipeline
from src.grok_llm import GroqLLM

load_dotenv()

# ---------------------------------------------------------------------------
# Pipeline initialisation (done once at startup)
# ---------------------------------------------------------------------------

def build_pipeline():
    api_key = os.environ.get("GROQ_API_KEY") or groq_api_key
    embedding_manager = EmbeddingManager(model_name=embedding_model_name)
    vector_store = VectorStore(
        collection_name=vector_db_name,
        persist_directory="data/vector_store"
    )
    retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager)
    llm = GroqLLM(model_name="gemma2-9b-it", api_key=api_key)
    return RAGPipeline(retriever=retriever, llm=llm)


pipeline: RAGPipeline | None = None
init_error: str | None = None

try:
    pipeline = build_pipeline()
except Exception as exc:
    init_error = str(exc)

# ---------------------------------------------------------------------------
# Query handler
# ---------------------------------------------------------------------------

def query(
    user_message: str,
    history: list,
    top_k: int,
    min_score: float,
    show_sources: bool,
    summarize: bool,
):
    if not user_message.strip():
        return history, history, ""

    if pipeline is None:
        error_msg = f"Pipeline failed to initialise: {init_error}"
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": error_msg})
        return history, history, ""

    try:
        result = pipeline.query_formatted(
            question=user_message,
            top_k=top_k,
            min_score=min_score,
            summarize=summarize,
        )
    except Exception as exc:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": f"Error: {exc}"})
        return history, history, ""

    answer = result["answer"]

    # Build sources panel text
    sources_md = ""
    if show_sources and result.get("sources"):
        lines = ["**Sources retrieved:**\n"]
        for i, src in enumerate(result["sources"], 1):
            score_pct = f"{src['score'] * 100:.1f}%"
            lines.append(
                f"**[{i}]** `{src['source']}` — page {src['page']} — score {score_pct}\n"
                f"> {src['preview']}\n"
            )
        sources_md = "\n".join(lines)

    if summarize and result.get("summary"):
        answer = f"{answer}\n\n---\n**Summary:** {result['summary']}"

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    return history, history, sources_md


def clear_chat():
    if pipeline:
        pipeline.history.clear()
    return [], [], ""

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="RAG Query UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 RAG Pipeline — Query Interface")
    gr.Markdown(
        "Ask questions about the documents in the knowledge base. "
        "Adjust retrieval settings in the sidebar."
    )

    with gr.Row():
        # ── Left column: chat ──────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=520,
                type="messages",
                show_copy_button=True,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Type your question and press Enter…",
                    show_label=False,
                    scale=5,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

        # ── Right column: settings + sources ──────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### ⚙️ Retrieval Settings")
            top_k_slider = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Top-K documents"
            )
            min_score_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                label="Min similarity score"
            )
            show_sources_cb = gr.Checkbox(value=True, label="Show sources")
            summarize_cb = gr.Checkbox(value=False, label="Summarize answer")

            gr.Markdown("### 📄 Retrieved Sources")
            sources_box = gr.Markdown(value="*Sources will appear here after a query.*")

    # ── State ──────────────────────────────────────────────────────────────
    chat_state = gr.State([])

    # ── Wire up events ─────────────────────────────────────────────────────
    shared_inputs = [msg_box, chat_state, top_k_slider, min_score_slider, show_sources_cb, summarize_cb]
    shared_outputs = [chatbot, chat_state, sources_box]

    send_btn.click(query, inputs=shared_inputs, outputs=shared_outputs).then(
        lambda: "", outputs=msg_box
    )
    msg_box.submit(query, inputs=shared_inputs, outputs=shared_outputs).then(
        lambda: "", outputs=msg_box
    )
    clear_btn.click(clear_chat, outputs=[chatbot, chat_state, sources_box])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
