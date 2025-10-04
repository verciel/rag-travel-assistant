# 🇮🇳 Indian Travel Query Assistant (RAG-based)

## 📌 Project Overview
This project is an **AI-powered Travel Query Answering System** designed to provide personalized travel recommendations about India.  
It leverages **Retrieval-Augmented Generation (RAG)** to combine curated tourism datasets with a Large Language Model (LLM) for accurate and contextual answers.

---

## 🎯 Objective
Tourists often face difficulty in finding reliable, context-specific travel recommendations (e.g., *“Best hill stations to visit in May”*).  
This project solves this problem by:
- Combining multiple Indian travel datasets  
- Enriching them with Wikipedia summaries  
- Creating a **vector database** for semantic search  
- Using an **LLM** to generate natural language answers  

---

## 🚀 Solution Architecture
1. **Data Collection** – Used three tourism datasets:  
   - `Top Indian Places to Visit.csv`  
   - `Expanded_Indian_Travel_Dataset.csv`  
   - `India-Tourism-Statistics-2021-Table-5.2.3.csv`  
   + enriched with Wikipedia summaries  

2. **Preprocessing & Embeddings** – Cleaned text, standardized schema, and generated embeddings with **SentenceTransformers**  

3. **Vector Indexing** – Built a **FAISS index** for fast similarity search  

4. **RAG Pipeline** –  
   - User query → converted into embedding  
   - FAISS → retrieves top relevant travel descriptions  
   - Context + query → passed to **Groq LLM (Llama-3 models)**  
   - LLM → generates a contextual travel recommendation  

5. **Interface** –  
   - **CLI mode** (terminal-based Q&A)  
   - **Gradio Web UI** (interactive user interface)  

---

## 🧠 Tech Stack
- **Python** (Pandas, NumPy)  
- **SentenceTransformers** (`all-MiniLM-L6-v2`) → embeddings  
- **FAISS** → vector database for similarity search  
- **Groq LLM** (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) → answer generation  
- **Gradio** → user interface  
- **Wikipedia REST API** → enrichment  

---

## 📊 Results
- Built a **unified knowledge base** of Indian travel destinations  
- Enabled **fast, semantic retrieval** using FAISS  
- Delivered **accurate and contextual travel recommendations** via Groq LLM  
- Supported **dual modes**: CLI and Web UI  

---

## 🌍 Applications
- Travel recommendation chatbots  
- Tourism websites and apps  
- AI assistants for government tourism portals  
- Smart travel planning systems  

---

## 🔮 Future Work
- Integrate live travel blogs, news, and weather data  
- Add multilingual support (Hindi, Tamil, Bengali, etc.)  
- Deploy as a **cloud-hosted web application** (Streamlit/HuggingFace)  

---

## 👨‍💻 Screenshots
<img width="1851" height="757" alt="Screenshot 2025-05-05 002005" src="https://github.com/user-attachments/assets/93cfdd9e-28b8-4fbd-8a0c-ed6de09ace4f" />
<img width="1855" height="791" alt="Screenshot 2025-05-05 002712" src="https://github.com/user-attachments/assets/c50d97d5-8066-496b-a82a-fdb97a19bee3" />

---

## 🛠️ Developer Notes (For Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/verciel/rag-travel-assistant.git
cd rag-travel-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Preprocess and Build Index
```bash
python src/data_preprocessing.py
python src/data_enrichment.py
python src/build_vector_index.py
```

### 4. Run the CLI
```bash
python main.py
```

### 5. Run the Gradio Web UI
```bash
python rag_interface.py
```
The UI will launch at `http://127.0.0.1:7860`.

