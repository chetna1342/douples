import json
import nltk
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Fixing random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load your data
with open('pcking.json') as json_file:
    intents = json.load(json_file)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize variables
patterns = []
labels = []
classes = []
words = []

for intent in intents:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        patterns.append(pattern)
        labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Lemmatize and remove stopwords
words = [lemmatizer.lemmatize(w.lower()) for w in words]
words = [word for word in words if word not in stop_words]
words = sorted(list(set(words)))

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# Create training data
training_sentences = []
for pattern in patterns:
    sentence_words = nltk.word_tokenize(pattern)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words if w not in stop_words]
    training_sentences.append(sentence_words)

# Create word index for tokenization
word_index = {word: i+1 for i, word in enumerate(words)}

# Convert words in training sentences to indexes and pad them
training_sentences = [
    [word_index.get(w, 0) for w in sentence] for sentence in training_sentences
]
# Use SentenceTransformer for embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
X = embedding_model.encode([" ".join([words[idx-1] for idx in sentence if idx > 0]) for sentence in training_sentences])
  # Generate embeddings for training sentences

# Ensure all sequences are of the same length (padding)
max_len = max([len(sentence) for sentence in training_sentences])
training_sentences = pad_sequences(training_sentences, padding='post', maxlen=max_len)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(training_sentences, labels, test_size=0.2, random_state=42)

# Define model
model = Sequential()
model.add(Embedding(input_dim=len(words)+1, output_dim=128, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))  # Dropout to avoid overfitting
model.add(LSTM(64))
model.add(Dropout(0.2))  # Dropout to avoid overfitting
model.add(Dense(32, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))  # Softmax for multi-class classification

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model.fit(np.array(X_train), np.array(y_train), epochs=50, batch_size=8, validation_data=(np.array(X_test), np.array(y_test)), verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(np.array(X_test), np.array(y_test), verbose=1)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

model.save('LSTM_Model_PCKING.keras')
pickle.dump(label_encoder, open('label_encoder_pcking.pkl', 'wb'))

