# ChatGPT User Reviews Classification using RNN and Word Embeddings
![GitHub](https://img.shields.io/badge/Python-3.8%2B-blue)
![GitHub](https://img.shields.io/badge/Library-TensorFlow%2FKeras-orange)
![GitHub](https://img.shields.io/badge/License-MIT-green)

##  Project Description 📝
This project implements a complete sentiment classification workflow on a real-world dataset of ChatGPT user reviews. We preprocess the text data, convert it into padded integer sequences via word embeddings, and train two neural architectures—SimpleRNN and LSTM—to compare performance on unseen test data.

##  Dataset 📂
- **Filename:** `chatgpt_reviews.csv`  
- **Total reviews:** 2,292  
  - Positive: 1,028 (45%)  
  - Negative: 1,264 (55%)  
- **Columns:**  
  - `review_text` – the raw user review  
  - `sentiment` – label (“Positive” or “Negative”)


## Objectives 🎯
- Preprocess raw text data for NLP tasks.
- Train and compare RNN and LSTM models for sentiment classification.
- Achieve high accuracy in predicting user review sentiments.
- Analyze the impact of hyperparameters (e.g., train-test split ratio, padding length).

## Technologies Used 🛠️
- **Programming Language**: Python
- **Libraries**:
  - `TensorFlow/Keras` for model building.
  - `NLTK` for text preprocessing.
  - `pandas` for data handling.
  - `scikit-learn` for data splitting.
- **Tools**: Jupyter Notebook/Google Colab.


## Machine Learning Algorithms 🤖
| Model     | Description                                                                |
|-----------|----------------------------------------------------------------------------|
| SimpleRNN | Standard recurrent layer with `units=…` for sequence modeling.             |
| LSTM      | Long Short-Term Memory layer to capture long-range dependencies in text.   |


## Getting Started 🚀

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

