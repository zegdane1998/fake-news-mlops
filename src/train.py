import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

def train():
    df = pd.read_csv('data/processed/gossipcop_cleaned.csv')
    X, y = df['clean_title'].astype(str), df['label']

    # Tokenization
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X)
    X_seq = pad_sequences(tokenizer.texts_to_sequences(X), maxlen=50)

    # CNN Architecture
    model = models.Sequential([
        layers.Embedding(10000, 128, input_length=50),
        layers.Conv1D(128, 5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), # Engineering best practice
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_seq, y, epochs=3, validation_split=0.2)
    model.save('models/cnn_v1.h5')
    print("Model saved to models/cnn_v1.h5")

if __name__ == "__main__":
    train()