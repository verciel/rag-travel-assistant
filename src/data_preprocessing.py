import pandas as pd
from sentence_transformers import SentenceTransformer

# Load datasets
df1 = pd.read_csv("Expanded_Indian_Travel_Dataset.csv")
df2 = pd.read_csv("India-Tourism-Statistics-2021-Table-5.2.3.csv")
df3 = pd.read_csv("Top Indian Places to Visit.csv")

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

def clean_text(text):
    return str(text).replace("\n", " ").strip()

def standardize(df, mapping):
    df = df.rename(columns=mapping)
    for col in ["title", "location", "state", "content"]:
        if col not in df.columns:
            df[col] = None
    df["content"] = df["content"].apply(clean_text)
    df["tags"] = df["content"].apply(lambda x: extract_tags(str(x)))
    if "best_time_to_visit" not in df.columns:
        df["best_time_to_visit"] = "Unknown"
    if "source_url" not in df.columns:
        df["source_url"] = "Unknown"
    return df[["title", "location", "state", "content", "tags", "best_time_to_visit", "source_url"]]

# Define mappings (adjust column names if needed)
map1 = {
    "Place_Name": "title",
    "City": "location",
    "State": "state",
    "Description": "content"
}

map2 = {
    "Destination": "title",
    "City_Name": "location",
    "State_Name": "state",
    "Details": "content"
}

map3 = {
    "title": "title",
    "location": "location",
    "description": "content"
}

# Standardize
df1_clean = standardize(df1, map1)
df2_clean = standardize(df2, map2)
df3_clean = standardize(df3, map3)

# Merge
combined_df = pd.concat([df1_clean, df2_clean, df3_clean], ignore_index=True)
combined_df["id"] = ["place_" + str(i) for i in range(len(combined_df))]

# Generate embeddings
print("Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
combined_df["embedding"] = combined_df["content"].apply(lambda x: model.encode(str(x)).tolist())

# Save final dataset
combined_df.to_json("unified_travel_dataset.json", orient="records", indent=2)
print("Saved to unified_travel_dataset.json")
