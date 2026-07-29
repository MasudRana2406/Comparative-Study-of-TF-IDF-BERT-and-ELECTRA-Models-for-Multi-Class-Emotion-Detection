# Emotion Classification Using TF-IDF, BERT, and ELECTRA Models

## 📌 Project Overview

This project focuses on **multi-class emotion classification from text data** using Natural Language Processing (NLP) techniques. The objective is to automatically identify human emotions expressed in text by comparing traditional machine learning approaches with modern transformer-based deep learning models.

A traditional **Logistic Regression model with TF-IDF feature extraction** was implemented as a baseline model. Its performance was compared against two state-of-the-art transformer models:

- **Fine-tuned BERT (Bidirectional Encoder Representations from Transformers)**
- **Fine-tuned ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)**

The project demonstrates how transformer-based architectures improve contextual understanding and classification performance compared with conventional text classification methods.

---

## 🎯 Objectives

The main objectives of this project are:

- Perform exploratory data analysis on an emotion text dataset.
- Analyse emotion class distribution and data quality.
- Develop a traditional machine learning baseline using TF-IDF and Logistic Regression.
- Fine-tune transformer-based models for emotion classification.
- Compare model performance using standard evaluation metrics.
- Analyse the strengths and limitations of different NLP approaches.

---

## 📂 Dataset

The project uses the **Emotion Dataset** from Hugging Face:

**Dataset:** dair-ai/emotion

The dataset contains text samples labelled into six emotion categories:

| Label | Emotion |
|-------|---------|
| 0 | Sadness |
| 1 | Joy |
| 2 | Love |
| 3 | Anger |
| 4 | Fear |
| 5 | Surprise |

Dataset split:

| Split | Samples |
|------|---------|
| Training | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Data Processing
- Pandas
- NumPy

### Data Visualisation
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- Logistic Regression
- TF-IDF Vectorization

### Deep Learning & NLP
- PyTorch
- Hugging Face Transformers
- BERT
- ELECTRA

### Dataset Handling
- Hugging Face Datasets

---

# 🔬 Methodology

## 1. Data Exploration and Preprocessing

The following preprocessing steps were performed:

- Dataset inspection
- Class distribution analysis
- Missing value checking
- Duplicate text detection
- Text length analysis
- Minimal text cleaning:
  - Lowercase conversion
  - Removal of unnecessary whitespace

---

## 2. Baseline Model: Logistic Regression + TF-IDF

A traditional NLP pipeline was developed:

**Text Data → TF-IDF Features → Logistic Regression Classifier → Emotion Prediction**

TF-IDF was used to convert text into numerical features based on word importance.

Parameters:

- Maximum features: 10,000
- Stop words removal: Enabled
- Logistic Regression maximum iterations: 1,000

---

## 3. Transformer Models

### BERT

The pre-trained:

`bert-base-uncased`

model was fine-tuned for six-class emotion classification.

Configuration:

- Maximum sequence length: 128 tokens
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Optimiser: AdamW

---

### ELECTRA

The pre-trained:

`google/electra-base-discriminator`

model was fine-tuned using the same training configuration.

ELECTRA was selected because it provides efficient pre-training by learning to distinguish replaced tokens instead of predicting masked tokens.

---

# 📊 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| Logistic Regression + TF-IDF | 0.867 | 0.867 | 0.867 | 0.863 |
| Fine-tuned BERT | 0.928 | 0.930 | 0.928 | 0.926 |
| Fine-tuned ELECTRA | 0.929 | 0.931 | 0.929 | 0.930 |

---

# 📈 Results and Findings

The experimental results show that transformer-based models significantly outperform traditional machine learning methods.

Key findings:

- Logistic Regression achieved strong baseline performance using TF-IDF features.
- BERT improved classification performance by understanding contextual relationships between words.
- ELECTRA achieved the highest overall performance with the best F1-score.
- Transformer models are more effective for complex emotion recognition tasks because they capture semantic meaning rather than relying only on word frequency.

---

# 📌 Model Evaluation

Evaluation metrics used:

### Accuracy
Measures the overall percentage of correctly classified samples.

### Precision
Measures how many predicted emotion labels were correct.

### Recall
Measures how many actual emotion samples were correctly identified.

### F1-score
Provides a balance between precision and recall.

Additional evaluation:

- Classification report
- Confusion matrix analysis

---

# 📁 Project Structure

```
Emotion-Classification-NLP/
│
├── README.md
│
├── emotion_classification.ipynb
│
├── models/
│   ├── bert-emotion-final/
│   └── electra-emotion-model/
│
├── requirements.txt
│
└── results/
    ├── confusion_matrix.png
    └── model_comparison.png
```

---

# 🚀 How to Run the Project

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/emotion-classification-nlp.git
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run Notebook

Open:

```
emotion_classification.ipynb
```

Execute the notebook cells sequentially.

---

# 📦 Requirements

Main libraries:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
transformers
datasets
```

---

# 🔮 Future Improvements

Possible improvements include:

- Testing larger transformer models such as RoBERTa or DeBERTa.
- Applying data augmentation techniques.
- Using class balancing methods for minority emotions.
- Performing hyperparameter optimisation.
- Deploying the model as an API using FastAPI or Streamlit.
- Building a real-time emotion detection application.

---

# 👨‍💻 Author

**Masud Rana**

MSc Data Science Student  
Interested in Natural Language Processing, Artificial Intelligence, and Machine Learning.

---

# 📚 References

- Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*

- Clark et al. (2020). *ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.*

- Vaswani et al. (2017). *Attention Is All You Need.*

- Hugging Face Transformers Documentation  
https://huggingface.co/docs/transformers
