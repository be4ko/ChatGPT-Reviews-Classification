# ChatGPT User Reviews Classification using RNN and Word Embeddings

## 📝 Project Description
This project implements a complete sentiment classification workflow on a real-world dataset of ChatGPT user reviews. We preprocess the text data, convert it into padded integer sequences via word embeddings, and train two neural architectures—SimpleRNN and LSTM—to compare performance on unseen test data.

## 📂 Dataset
- **Filename:** `chatgpt_reviews.csv`  
- **Total reviews:** 2,292  
  - Positive: 1,028 (45%)  
  - Negative: 1,264 (55%)  
- **Columns:**  
  - `review_text` – the raw user review  
  - `sentiment` – label (“Positive” or “Negative”)


## 🎯 Objectives
- Demonstrate an end-to-end NLP pipeline for sentiment analysis  
- Compare performance between SimpleRNN and LSTM architectures  
- Showcase the impact of sequence length and train/test split ratio on model accuracy (bonus)


## 🛠️ Technologies Used
- **Python 3.8+**  
- **Keras / TensorFlow** – RNN & embedding layers  
- **NLTK** – tokenization & stopword removal  
- **pandas** – data loading & manipulation  
- **scikit-learn** – train/test splitting & label encoding  


## 🤖 Machine Learning Algorithms
| Model     | Description                                                                |
|-----------|----------------------------------------------------------------------------|
| SimpleRNN | Standard recurrent layer with `units=…` for sequence modeling.             |
| LSTM      | Long Short-Term Memory layer to capture long-range dependencies in text.   |


## 🚀 Getting Started

1. **Clone repo**  
   ```bash
   git clone https://github.com/<your-username>/chatgpt-reviews-classification.git
   cd chatgpt-reviews-classification 

2. **Install dependencies:**
  ``` bash
  pip install tensorflow nltk pandas numpy scikit-learn
```
3. **Run the Project**
``` bash
// to be written later :)
```

