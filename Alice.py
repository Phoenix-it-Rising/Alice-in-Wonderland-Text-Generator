"""
Author: Kiana Lang
Date: September 27, 2025
Course: CS492 - Machine Learning
Description: This script trains an LSTM model on the text of 'Alice in Wonderland' to generate new text.
             LSTM is used due to its strength in modeling sequential data and capturing long-term dependencies.
             The script includes preprocessing, model training, and text generation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess the dataset
def load_text(file_path):
    """Reads the text file and converts it to lowercase."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return text

def tokenize_text(text):
    """Splits the text into individual words."""
    return text.split()

def create_vocab(tokens):
    """Creates word-to-index and index-to-word mappings."""
    vocab = sorted(set(tokens))
    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    return word_to_id, id_to_word, vocab

def create_sequences(tokens, word_to_id, seq_length):
    """Generates sequences of words for training."""
    sequences = []
    for i in range(seq_length, len(tokens)):
        seq = tokens[i-seq_length:i+1]
        encoded = [word_to_id[word] for word in seq]
        sequences.append(encoded)
    return sequences

def prepare_data(sequences, vocab_size):
    """Splits sequences into input (X) and output (y) and one-hot encodes y."""
    sequences = np.array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    return X, y

def build_model(vocab_size, seq_length):
    """Defines and compiles the LSTM model."""
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=50))  # Removed deprecated input_length
    model.add(LSTM(100))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def generate_text(model, seed_text, word_to_id, id_to_word, seq_length, num_words):
    """Generates new text based on a seed input with randomness."""
    result = seed_text.split()
    for _ in range(num_words):
        encoded = [word_to_id.get(word, 0) for word in result[-seq_length:]]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        preds = model.predict(encoded, verbose=0)[0]
        pred_id = np.random.choice(len(preds), p=preds)  # Random sampling for varied output
        result.append(id_to_word[pred_id])
    return ' '.join(result)

def main():
    file_path = r"C:\Users\User\Downloads\4944029-75ba2243dd5ec2875f629bf5d79f6c1e4b5a8b46\4944029-75ba2243dd5ec2875f629bf5d79f6c1e4b5a8b46\alice_in_wonderland.txt"
    text = load_text(file_path)
    tokens = tokenize_text(text)
    word_to_id, id_to_word, vocab = create_vocab(tokens)
    vocab_size = len(vocab)
    seq_length = 10

    sequences = create_sequences(tokens, word_to_id, seq_length)
    X, y = prepare_data(sequences, vocab_size)

    model = build_model(vocab_size, seq_length)
    model.fit(X, y, epochs=5, batch_size=128)

    seed_text = 'the rabbit took a watch out of its waistcoat'
    generated = generate_text(model, seed_text, word_to_id, id_to_word, seq_length, 20)
    print("Generated Text:\n", generated)

if __name__ == "__main__":
    main()

