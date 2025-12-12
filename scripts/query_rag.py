# query_rag_api.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Load Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    raise ValueError("❌ ERROR: GROQ_API_KEY not found! Add it in your .env file.")

# Load FAISS index and metadata
try:
    index = faiss.read_index("embeddings/monuments_index.faiss")
    metadata = np.load("embeddings/metadata.npy", allow_pickle=True)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load FAISS or metadata: {e}")

# Load embedding model
print("🔄 Loading embedding model (SentenceTransformer)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedder Loaded!")


# -----------------------------------------------------------
# 📌 RETRIEVAL FUNCTION
# -----------------------------------------------------------
def retrieve(query, top_k=5):
    try:
        query_vec = model.encode([query]).astype("float32")
        D, I = index.search(query_vec, top_k)

        results = []
        for idx in I[0]:
            item = metadata[idx]
            monument_name = item.get("monument", "Unknown Monument")
            city_name = item.get("city", "Unknown City")
            results.append(f"{monument_name} ({city_name})")
        return results

    except Exception as e:
        print("❌ Retrieval failed!", e)
        return []


# -----------------------------------------------------------
# 📌 GROQ API CALL FUNCTION
# -----------------------------------------------------------
def query_groq(prompt_text):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",     # fast + cheap + high quality
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 350,
        "temperature": 0.6
    }

    response = requests.post(GROQ_ENDPOINT, json=data, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.status_code}\n{response.text}")

    return response.json()["choices"][0]["message"]["content"]


# -----------------------------------------------------------
# 📌 MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    print("\n✨ India Travel RAG System ✨")
    print("----------------------------------------")

    user_query = input("Enter your travel query: ")

    print("\n🔍 Retrieving monuments related to your query...")
    top_monuments = retrieve(user_query, top_k=5)

    if len(top_monuments) == 0:
        print("⚠️ No similar monuments found in FAISS index.")
        exit()

    print("✅ Retrieved:", top_monuments)

    context = "\n".join(top_monuments)

    prompt = f"""
You are an expert India travel itinerary planner.

Use ONLY the following context about monuments:

{context}

Now answer the user's question:

User Query: {user_query}

Provide a clear, accurate itinerary or travel guidance.
"""

    print("\n🧠 Generating itinerary using GROQ...")
    answer = query_groq(prompt)

    print("\n------------------------------")
    print("🧳 ✨ Generated Itinerary ✨ 🧳")
    print("------------------------------\n")
    print(answer)
    print("\n------------------------------")
