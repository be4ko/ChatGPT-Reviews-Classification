import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, \
    Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Load and preprocess data
df = pd.read_csv('chatgpt_reviews.csv')
df = df.dropna()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    # Keep contractions and negations which are important for sentiment
    text = re.sub(r'[^a-zA-Z\']', ' ', text)  # Keep apostrophes for contractions
    text = text.lower()

    # Replace common contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'re", " are", text)

    words = text.split()
    # Keep important sentiment words including negations
    important_words = {'not', 'no', 'nor', 'but', 'don', "don't", "doesn't", "didn't",
                       "couldn't", "shouldn't", "wouldn't", "haven't", "hasn't", "won't",
                       "can't", "never", "none", "nothing", "nowhere", "hardly", "barely",
                       "scarcely", "rarely", "seldom", "neither", "nor"}
    words = [word for word in words if word not in stop_words or word in important_words]
    return ' '.join(words)


df['Review'] = df['Review'].apply(clean_text)
df['label'] = df['label'].map({'POSITIVE': 1, 'NEGATIVE': 0})

# Check class distribution
print("Class distribution:")
print(df['label'].value_counts(normalize=True))

# Split data with stratification to maintain class distribution
X = df['Review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Tokenization with more nuanced parameters
max_features = 10000  # Limit vocabulary size to most frequent words
tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Analyze sequence lengths to choose appropriate padding
lengths = [len(seq) for seq in X_train_seq]
print(f"Average sequence length: {np.mean(lengths)}")
print(f"Median sequence length: {np.median(lengths)}")
print(f"95th percentile length: {np.percentile(lengths, 95)}")

# Use a more appropriate max_length based on data distribution
max_length = int(np.percentile(lengths, 95))  # Using 95th percentile instead of maximum
print(f"Using max_length: {max_length}")

vocab_size = min(len(tokenizer.word_index) + 1, max_features)
print(f"Vocabulary size: {vocab_size}")

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')


# Define improved models
def create_improved_rnn_model():
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        Bidirectional(SimpleRNN(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(SimpleRNN(64)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_improved_lstm_model():
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



# Train and evaluate improved SimpleRNN
print("\nTraining Improved Bidirectional RNN Model:")
rnn_model = create_improved_rnn_model()
rnn_history = rnn_model.fit(
    X_train_pad, y_train,
    epochs=15,
    batch_size=64,  # Smaller batch size for better generalization
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)
rnn_acc = rnn_model.evaluate(X_test_pad, y_test)[1]
print(f"Improved RNN Test Accuracy: {rnn_acc:.4f}")

# Train and evaluate improved LSTM
print("\nTraining Improved Bidirectional LSTM Model:")
lstm_model = create_improved_lstm_model()
lstm_history = lstm_model.fit(
    X_train_pad, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)
lstm_acc = lstm_model.evaluate(X_test_pad, y_test)[1]
print(f"Improved LSTM Test Accuracy: {lstm_acc:.4f}")


# Print final comparison
print("\nModel Accuracy Comparison:")
print(f"Improved RNN: {rnn_acc:.4f}")
print(f"Improved LSTM: {lstm_acc:.4f}")
