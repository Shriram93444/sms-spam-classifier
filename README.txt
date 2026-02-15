# SMS Spam Classifier – Machine Learning Project 
An end-to-end Machine Learning project that detects whether an SMS message is Spam or Not Spam
 using Natural Language Processing (NLP).

## Project Overview

Built an **SMS spam/ham classifier** using the classic UCI SMS Spam Collection dataset (~5,574 messages).  
Demonstrates a full beginner ML pipeline: data cleaning → feature extraction → model training → evaluation.

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn

## Results

Trained and compared two models:

| Model                     | Accuracy | Notes                                      |
|---------------------------|----------|--------------------------------------------|
| Multinomial Naive Bayes   |   97.1%  | Strong baseline — fast & effective on text |
| Logistic Regression       |   96.9%  | Slightly lower here; good interpretability |

Confusion Matrix (Logistic Regression)
![Confusion Matrix - Logistic Regression](assets/confusion_matrix.png)

## Key Learnings

- Text cleaning (lowercase, remove punctuation/numbers, stop words) makes a big difference  
- TF-IDF vectorization is simple yet powerful for bag-of-words problems  
- Naive Bayes excels on text classification due to its probabilistic nature  
- Precision > recall in spam detection (avoid blocking legitimate messages)  
- Model comparison helps understand trade-offs

## How to Run

```bash
# 1. Clone repo
git clone https://github.com/Shriram93444/sms-spam-classifier
cd sms-spam-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebook
Spam_Classifier.ipynb
