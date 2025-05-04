# بسم الله الرحمن الرحيم

import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, \
    Conv1D, MaxPooling1D
from tensorflow.keras.regularizers import l2



# nltk.download('stopwords')
# print(stopwords.words('english'))


df = pd.read_csv('chatgpt_reviews.csv')
df = df.dropna()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    important_stopwords = {'not', 'no', 'nor', 'but', 'don', "don't", "doesn't", "didn't",
                       "couldn't", "shouldn't", "wouldn't", "haven't", "hasn't", "won't",
                       "can't", "never", "none", "nothing", "nowhere", "hardly", "barely",
                       "scarcely", "rarely", "seldom", "neither", "nor"}
    words = [word for word in words if word not in stop_words or word in important_stopwords]
    return ' '.join(words)

df['Review'] = df['Review'].apply(clean_text)
df['label'] = df['label'].map({'POSITIVE': 1, 'NEGATIVE': 0})


df.to_csv('clean_reviews.csv', index=False)
print("Cleaned data saved to 'clean_reviews.csv'")

# Split data
X = df['Review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = max(len(seq) for seq in X_train_seq)
vocab_size = len(tokenizer.word_index) + 1


X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')


def rnn_model():
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        Bidirectional(SimpleRNN(64, return_sequences=True)),
        Bidirectional(SimpleRNN(64)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def lstm_model():
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

rnn_model = rnn_model()
rnn_model.fit(X_train_pad, y_train, epochs=10, batch_size=128, validation_split=0.1)
rnn_acc = rnn_model.evaluate(X_test_pad, y_test, verbose=0)[1]
print(f"SimpleRNN Test Accuracy: {rnn_acc:.4f}")

lstm_model = lstm_model()
lstm_model.fit(X_train_pad, y_train, epochs=10, batch_size=128, validation_split=0.1)
lstm_acc = lstm_model.evaluate(X_test_pad, y_test, verbose=0)[1]
print(f"LSTM Test Accuracy: {lstm_acc:.4f}")


# Hyperparameter report
print("\nModel Summaries:")
print("SimpleRNN:")
rnn_model.summary()
print("\nLSTM:")
lstm_model.summary()