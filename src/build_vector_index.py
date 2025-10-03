import json
import faiss
import numpy as np
import pickle

# Load your cleaned dataset
with open("unified_travel_dataset_filled.json", "r") as f:
    data = json.load(f)

# Extract embeddings and metadata
embeddings = []
metadata = []

for entry in data:
    emb = entry.get("embedding")
    if emb:
        embeddings.append(np.array(emb).astype("float32"))
        metadata.append({
            "title": entry["title"],
            "location": entry["location"],
            "state": entry["state"],
            "content": entry["content"],
            "tags": entry["tags"],
            "url": entry["source_url"]
        })

# Build FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save FAISS index + metadata
faiss.write_index(index, "travel_index.faiss")
with open("travel_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("FAISS index and metadata saved!")
