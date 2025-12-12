# Script to build FAISS index

import faiss
import numpy as np

embeddings = np.load("embeddings/monuments_embeddings.npy")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "embeddings/monuments_index.faiss")
print(f"Indexed {index.ntotal} monuments")
