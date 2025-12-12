# Script to create embeddings
#downloads the SentenceTransformer model needed for generating embeddings and prepares everything RAG system needs.
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Load JSON
with open("data/india_monuments.json", "r", encoding="utf-8") as f:
    monuments_data = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = []
metadata = []

for city in monuments_data:
    for monument in city['monuments']:
        text = f"{monument['name']}: {monument.get('notes','')} Entry fee: {monument.get('entry_fee', monument.get('entry_fee_indian','N/A'))}"
        texts.append(text)
        metadata.append({"city": city['city'], "monument": monument['name']})

embeddings = model.encode(texts)
np.save("embeddings/monuments_embeddings.npy", embeddings)
np.save("embeddings/metadata.npy", metadata)
