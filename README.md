# Earnings Call Sentiment Analysis with FinBERT

A financial NLP project that analyzes earnings call transcripts using FinBERT, a finance-specific transformer model based on BERT. The project evaluates whether management sentiment during earnings calls can predict short-term post-earnings stock returns.

---

# Overview

This project applies Natural Language Processing (NLP) and event-study methodology to earnings call transcripts.

The workflow:

```text
Earnings Call Transcript
        ↓
FinBERT Sentiment Analysis
        ↓
Sentiment Score Generation
        ↓
Event Study on Stock Returns
        ↓
Strategy Backtest & Analysis
```

The project compares:

- Transformer-based financial sentiment (FinBERT)
- Traditional financial dictionary sentiment (Loughran-McDonald)

---

# Features

- FinBERT sentiment classification
- Earnings transcript processing
- Financial NLP pipeline
- Event-study framework
- Forward return analysis (1–5 days)
- Long/short sentiment strategy
- Visualization of sentiment vs returns
- Dictionary-based sentiment benchmark
- Clean modular Python structure

---

# Technologies Used

```text
Python
Transformers (Hugging Face)
PyTorch
Pandas
NumPy
Matplotlib
Scikit-learn
yFinance
```

---

# FinBERT Model

This project uses:

```text
ProsusAI/finbert
```

FinBERT is a finance-domain adaptation of BERT trained on large financial corpora and fine-tuned for financial sentiment classification.

The model predicts:

```text
positive
negative
neutral
```

Sentiment score:

```python
sentiment_score = positive_probability - negative_probability
```

---

# Project Structure

```text
earnings-call-sentiment-analyzer/
│
├── README.md
├── requirements.txt
├── run.py
│
├── src/
│   └── earnings_sentiment/
│       ├── __init__.py
│       ├── data.py
│       ├── finbert.py
│       ├── dictionary.py
│       ├── event_study.py
│       ├── backtest.py
│       └── plotting.py
│
├── notebooks/
│   └── finbert_analysis.ipynb
│
├── reports/
│   ├── figures/
│   └── results/
│
└── tests/
    ├── test_finbert.py
    ├── test_event_study.py
    └── test_dictionary.py
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/earnings-call-sentiment-analyzer.git
cd earnings-call-sentiment-analyzer
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

## Windows

```bash
venv\Scripts\activate
```

## Mac/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the complete pipeline:

```bash
python run.py
```

Or run inside Jupyter Notebook:

```bash
jupyter notebook
```

---

# Example Usage

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

text = """
Revenue exceeded expectations and guidance was raised.
Demand remains strong across all business segments.
"""

result = classifier(text)

print(result)
```

Example output:

```text
[
  {
    'label': 'positive',
    'score': 0.998
  }
]
```

---

# Event Study

For each earnings call:

1. Transcript sentiment is calculated
2. Post-earnings returns are measured
3. Forward returns are analyzed over:

```text
1 Day
2 Day
3 Day
4 Day
5 Day
```

The project tests whether:

```text
Positive sentiment → Positive future returns
Negative sentiment → Negative future returns
```

---

# Long/Short Strategy

The strategy:

```text
Long companies with positive earnings sentiment
Short companies with negative earnings sentiment
```

Performance metrics include:

- Average return
- Correlation with future returns
- Strategy equity curve
- Sentiment bucket analysis

---

# Sample Output

| Ticker | Sentiment | Score | 5D Return |
|--------|-----------|--------|-----------|
| AAPL | Positive | 0.81 | 3.2% |
| TSLA | Negative | -0.67 | -2.4% |
| NVDA | Positive | 0.92 | 5.8% |

---

# Research Papers

## FinBERT

Yang, Yi, Mark Christopher Siy Uy, and Allen Huang.

**FinBERT: A Pretrained Language Model for Financial Communications**  
2020

Paper:

```text
https://arxiv.org/abs/2006.08097
```

---

## Loughran-McDonald Dictionary

Loughran, Tim, and Bill McDonald.

**When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks**  
Journal of Finance, 2011

---

# Why This Project Matters

Financial markets react not only to numerical earnings results but also to management tone, guidance, and language.

This project demonstrates how transformer-based NLP models can extract sentiment signals from financial text and test their predictive power in equity markets.

---

# Skills Demonstrated

- Financial Machine Learning
- NLP for Finance
- Transformer Models
- Event Studies
- Quantitative Research
- Python Package Design
- Data Analysis
- Backtesting

---

# Future Improvements

- Real transcript scraping pipeline
- SEC filing integration
- Multi-factor models
- Portfolio optimization
- Walk-forward validation
- Intraday event studies
- Fine-tuning FinBERT on earnings calls

---

# Disclaimer

This project is for educational and research purposes only.  
It is not financial advice or a live trading system.

---
