import os
import json
import pickle
import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load FAISS index and metadata
index = faiss.read_index("travel_index.faiss")
with open("travel_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Search the FAISS index for top matches
def get_top_k_contexts(query, k=3):
    query_embedding = model.encode([query]).astype("float32")
    D, I = index.search(query_embedding, k)
    return "\n\n".join([metadata[i]["content"] for i in I[0]])

# Ask Groq LLM with context
def ask_llm(query):
    context = get_top_k_contexts(query)
    prompt = f"""You are a helpful Indian travel assistant. Use the following information to answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  #  Replace with another if needed
        messages=[
            {"role": "system", "content": "You are a helpful Indian travel assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content

# Entry point
if __name__ == "__main__":
    query = input("Ask a travel question: ")
    print("\n💡 Answer:\n")
    print(ask_llm(query))