import json
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
import random
import pickle
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

with open('surela.json') as json_file:
    intents = json.load(json_file)

# Preprocess data
lemmatizer = nltk.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

training_sentences = []
training_labels = []
labels = []

for intent in intents:
    for pattern in intent['patterns']:
        words = nltk.word_tokenize(pattern)
        words = [lemmatizer.lemmatize(w.lower()) for w in words if w.lower() not in stop_words]
        training_sentences.append(" ".join(words))
        training_labels.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(training_labels)

# Use SentenceTransformer for embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
X = embedding_model.encode(training_sentences)  # Generate embeddings for training sentences

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),  # Input shape matches the embedding size
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
labels = label_encoder.classes_  # Get the actual labels
print(classification_report(y_test, y_pred_classes, target_names=labels, labels=np.unique(y_test)))
print("Accuracy:", accuracy_score(y_test, y_pred_classes))

model.save('LSTM_Model.keras')
pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))
