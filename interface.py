import os
import json
import pickle
import faiss
import gradio as gr
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load JSON data
with open("unified_travel_dataset_filled.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [doc["content"] for doc in data]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents).astype("float32")

if not os.path.exists("travel_index.faiss"):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "travel_index.faiss")
    with open("travel_metadata.pkl", "wb") as f:
        pickle.dump(data, f)

index = faiss.read_index("travel_index.faiss")
with open("travel_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_top_k_contexts(query, k=3):
    query_embedding = model.encode([query]).astype("float32")
    _, I = index.search(query_embedding, k)
    return "\n\n".join([metadata[i]["content"] for i in I[0]])

def ask_llm(query):
    context = get_top_k_contexts(query)
    prompt = f"""You are a helpful Indian travel assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful Indian travel assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()

# Enhanced UI
with gr.Blocks(title="Indian Travel RAG Assistant") as demo:
    gr.Markdown("# 🧳 Incredible India Travel Assistant")
    gr.Markdown("Ask about Indian destinations, culture, food, weather, and more. Built using RAG + Groq LLM.")
    
    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="Ask a travel question",
                placeholder="e.g., What are the best hill stations to visit in May?",
                lines=2
            )
            submit_btn = gr.Button("Get Answer 🎯", variant="primary")
        with gr.Column(scale=8):
            output = gr.Textbox(label="💡 Answer", lines=10)

    submit_btn.click(fn=ask_llm, inputs=query_input, outputs=output)

if __name__ == "__main__":
    demo.launch()
