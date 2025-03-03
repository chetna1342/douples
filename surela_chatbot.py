import os
import json
import numpy as np
import pickle
import random
from tensorflow.keras.models import load_model
import faiss
from Surela_faiss_index import load_faiss_index
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load models and required files
lstm_model = load_model('LSTM_Model.keras')
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
# Ensure you have the correct vectorizer or remove this line if not needed
# tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Load FAISS index & labels
faiss_index, intent_labels = load_faiss_index()

# Load responses for intent-based reply
with open('surela.json') as json_file:
    responses = json.load(json_file)

# Load embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Set up GROQ API token
GROQ_API_TOKEN = os.getenv("GROQ_API_TOKEN", "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")
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

def get_intent(user_input, faiss_index, labels, embedding_model, threshold=0.7):
    input_vector = embedding_model.encode([user_input]).astype('float32')

    # Search in FAISS
    D, I = faiss_index.search(input_vector, k=1)
    distance = D[0][0]
    index = I[0][0]

    # Validate index
    if 0 <= index < len(labels) and distance < threshold:
        return labels[index]

    return None  # No valid match

def filter_response(user_input):
    # Use SentenceTransformer to generate embeddings for the user input
    input_vector = embedding_model.encode([user_input]).astype('float32')

    # Reshape the input_vector to match the LSTM input shape
    input_vector = np.reshape(input_vector, (1, -1))  # Ensure it's 2D

    prediction = lstm_model.predict(input_vector, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]

def fallback_to_gemini(user_input):
    response = llm.invoke(user_input)
    return response.content

def chatbot():
    print("Chatbot is ready! Type 'exit' to quit.")

    while True:
        user_input = input("You: ").strip().lower()
        if user_input == 'exit':
            break

        try:
            intent = get_intent(user_input, faiss_index, intent_labels, embedding_model)

            if intent:
                # Filter the response based on LSTM model
                filtered_response = filter_response(user_input)

                if filtered_response == "competitor":
                    print("Filtered out competitor-related response.")
                    continue  # Skip to the next input if a competitor response is detected

                response = next(
                    (random.choice(intent_obj['responses']) for intent_obj in responses if intent_obj['tag'] == intent),
                    None
                )

                if not response:
                    print("No predefined response found. Falling back to Gemini.")
                    response = fallback_to_gemini(user_input)
            else:
                print("No intent match found, falling back to LLM.")
                response = fallback_to_gemini(user_input)

            print(f"Bot: {response}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chatbot()
