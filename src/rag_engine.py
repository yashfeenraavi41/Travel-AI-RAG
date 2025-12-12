#loads FAISS + Groq model
#RAG logic: retrieval + prompt building + Groq call
# rag_engine.py
import os
import faiss
import numpy as np
import json
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Config: update if your endpoint differs
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths (assumes your structure)
EMBED_DIR = os.path.join(BASE_DIR, "..", "embeddings")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "india_monuments.json")

# Load metadata, index and embeddings
_index_path = os.path.join(EMBED_DIR, "monuments_index.faiss")
_metadata_path = os.path.join(EMBED_DIR, "metadata.npy")

if not os.path.exists(_index_path) or not os.path.exists(_metadata_path):
    raise FileNotFoundError("FAISS index or metadata not found in embeddings/ - run create_embeddings/build_index first.")

index = faiss.read_index(_index_path)
metadata = np.load(_metadata_path, allow_pickle=True)

# Load full data JSON so we can include detailed monument info (entry fees, timings, notes)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    monuments_json = json.load(f)

# Build a lookup: (city, monument_name) -> details_text
_monument_lookup = {}
for city_obj in monuments_json:
    city_name = city_obj.get("city")
    for m in city_obj.get("monuments", []):
        name = m.get("name")
        # Build a short description text for RAG context
        parts = [f"Name: {name}", f"City: {city_name}"]
        if "entry_fee" in m:
            parts.append(f"Entry fee: {m.get('entry_fee')}")
        elif "entry_fee_indian" in m or "entry_fee_foreign" in m:
            parts.append(f"Entry fees - Indian: {m.get('entry_fee_indian','N/A')}, Foreign: {m.get('entry_fee_foreign','N/A')}")
        if m.get("timings"):
            parts.append(f"Timings: {m.get('timings')}")
        if m.get("notes"):
            parts.append(f"Notes: {m.get('notes')}")
        _monument_lookup[(city_name.lower(), name.lower())] = " | ".join(parts)

# Sentence embedder used both for retrieval
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Map interests to India-specific categories (same mapping as your old backend)
INDIA_INTEREST_MAP = {
    "Temples & Shrines": "temples, shrines, religious sites",
    "Forts & Palaces": "historic forts, palaces, royal heritage",
    "Cultural Heritage": "cultural heritage, traditional arts",
    "Traditional Food": "local cuisine, street food, traditional dishes",
    "Museums & Art Galleries": "museums, art galleries, exhibitions"
}

BUDGET_MAP = {
    "Budget Friendly": "Focus on attractions with low entry fees and local experiences (under ₹2,000/day).",
    "Moderate": "Mix of popular attractions and comfortable options (₹2,000–₹5,000/day).",
    "Luxury Experience": "Include premium hotels, experiences, and guided tours (above ₹5,000/day)."
}

def retrieve(query, city_filter=None, top_k=6):
    """
    Retrieve top_k monuments from FAISS using semantic similarity.
    If city_filter is provided, results are filtered by city (best-effort).
    """
    qvec = embedder.encode([query]).astype("float32")
    D, I = index.search(qvec, top_k * 3)  # retrieve a few extra so we can filter by city
    results = []
    for idx in I[0]:
        meta = metadata[idx].item() if hasattr(metadata[idx], "item") else metadata[idx]
        # metadata object format created earlier: {"city": city, "monument": monument_name}
        city_name = meta.get("city", "")
        monument_name = meta.get("monument", "")
        results.append((city_name, monument_name))
    # If city_filter specified, sort to prefer same city
    if city_filter:
        city_filter_l = city_filter.lower()
        filtered = [m for m in results if m[0].lower() == city_filter_l]
        if len(filtered) >= top_k:
            results = filtered[:top_k]
        else:
            # include same-city ones first, then others
            same = [m for m in results if m[0].lower() == city_filter_l]
            others = [m for m in results if m[0].lower() != city_filter_l]
            combined = same + others
            results = combined[:top_k]
    else:
        results = results[:top_k]

    # convert to descriptive texts using lookup
    texts = []
    for city_name, monument_name in results:
        key = (city_name.lower(), monument_name.lower())
        text = _monument_lookup.get(key)
        if not text:
            # fallback minimal text
            text = f"Name: {monument_name} | City: {city_name}"
        texts.append(text)
    return texts

def call_groq_system(prompt_text):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment.")
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful Indian travel assistant."},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 1200
    }
    resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # compatible with Groq OpenAI-style returns
    return data["choices"][0]["message"]["content"]

def generate_rag_itinerary(city, location=None, trip_duration="1-day", budget="Moderate", interests=[]):
    # Build interest string
    interest_texts = []
    for i in interests:
        mapped = INDIA_INTEREST_MAP.get(i)
        if mapped:
            interest_texts.append(mapped)
    interest_str = ", ".join(interest_texts) if interest_texts else ", ".join(interests)

    location_str = f"User is currently at latitude/longitude: {location}. " if location else ""

    budget_str = BUDGET_MAP.get(budget, "")

    # Build retrieval query
    retrieval_query = f"{city} monuments, {interest_str} {budget_str}"
    retrieved = retrieve(retrieval_query, city_filter=city, top_k=6)
    context = "\n".join(retrieved)

    # Compose prompt (mirrors your Django prompt but includes context)
    prompt = f"""
You are an expert India travel planner AI.
{location_str}
Use ONLY the monuments and details provided in the CONTEXT below to prepare the itinerary.

Context:
{context}

Create a detailed day-wise travel itinerary for a trip in {city}.
Trip Duration: {trip_duration}
Interests: {interest_str}
Budget guidance: {budget_str}

Include:
  - Famous monuments (use only those in context when possible)
  - Suggested timings for visiting attractions
  - Local food and cultural experiences
  - Day-wise schedule

Respond in plain text with 'Day 1:', 'Day 2:', etc. Also list the monument entry fees and visiting hours where available from the context.
"""

    # Call Groq
    itinerary_text = call_groq_system(prompt)
    return itinerary_text
