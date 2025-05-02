# بسم الله الرحمن الرحيم
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import re
# import numpy as np
# import nltk

#nltk.download('stopwords')
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


