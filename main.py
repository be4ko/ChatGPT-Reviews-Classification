# بسم الله الرحمن الرحيم

import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import re
from keras_preprocessing.text import Tokenizer
# import os
# os.environ["KERAS_BACKEND"] = "jax"


# nltk.download('stopwords')
# print(stopwords.words('english'))


df = pd.read_csv('chatgpt_reviews.csv')
df = df.dropna()

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['Review'] = df['Review'].apply(clean_text)
df['label'] = df['label'].map({'POSITIVE': 1, 'NEGATIVE': 0})


df.to_csv('clean_reviews.csv', index=False)
print("Cleaned data saved to 'clean_reviews.csv'")

# Split data
X = df['Review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = max(len(seq) for seq in X_train_seq)
vocab_size = len(tokenizer.word_index) + 1

print(vocab_size)
