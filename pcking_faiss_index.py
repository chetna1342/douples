import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load intents from surela.json
with open('pcking.json') as json_file:
    intents = json.load(json_file)

# Initialize Sentence Transformer for embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def create_faiss_index(intents):
    vectors = []
    labels = []

    for intent in intents:
        for pattern in intent['patterns']:
            vector = embedding_model.encode(pattern)
            vectors.append(vector)
            labels.append(intent['tag'])

    vectors = np.array(vectors).astype('float32')

    # FAISS index creation
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, labels

def save_faiss_index(index, labels, filename="faiss_index.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((index, labels), f)

def load_faiss_index(filename="faiss_index.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Train FAISS & Save
    faiss_index, intent_labels = create_faiss_index(intents)
    save_faiss_index(faiss_index, intent_labels)
    print("✅ FAISS index created & saved successfully!")
