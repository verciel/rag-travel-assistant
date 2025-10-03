import json
import requests
from sentence_transformers import SentenceTransformer

def get_wikipedia_summary(query):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '%20')}"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        return data.get("title", query), data.get("extract", "")
    return query, ""

def extract_tags(text):
    tags = []
    keywords = {
        "beach": ["beach", "coast", "sea", "sand"],
        "hill": ["hill", "mountain", "trek", "valley"],
        "heritage": ["temple", "fort", "palace", "ruins"],
        "wildlife": ["forest", "safari", "tiger", "park"],
        "spiritual": ["ashram", "meditation", "yoga", "spiritual"]
    }
    for tag, keys in keywords.items():
        if any(k in text.lower() for k in keys):
            tags.append(tag)
    return tags

# Load your dataset
with open("unified_travel_dataset.json", "r") as f:
    data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Fill missing fields using Wikipedia
for item in data:
    if not item.get("content") or item["content"] in [None, "", "None"]:
        state = item.get("state", "")
        if not state:
            continue
        title, summary = get_wikipedia_summary(state)
        item["title"] = title
        item["location"] = state
        item["content"] = summary
        item["tags"] = extract_tags(summary)
        item["best_time_to_visit"] = "Unknown"
        item["source_url"] = f"https://en.wikipedia.org/wiki/{state.replace(' ', '_')}"
        item["embedding"] = model.encode(summary).tolist()

# Save the updated dataset
with open("unified_travel_dataset_filled.json", "w") as f:
    json.dump(data, f, indent=2)

print("Done. Saved to unified_travel_dataset_filled.json")
