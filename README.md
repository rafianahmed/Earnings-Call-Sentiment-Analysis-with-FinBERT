# Earnings-Call-Sentiment-Analysis-with-FinBERT

This project analyzes earnings call transcripts using FinBERT, a finance-tuned BERT model for sentiment classification. It scores each transcript as positive, negative, or neutral, then tests whether sentiment predicts short-term post-earnings stock returns.

## Project Overview

The goal is to evaluate whether earnings call sentiment contains useful information for predicting stock returns over the next 1 to 5 trading days.

The project compares two sentiment approaches:

1. **FinBERT transformer sentiment model**
2. **Loughran-McDonald financial dictionary baseline**

The final output includes sentiment scores, event-study returns, strategy performance, and comparison plots.

## Methodology

### 1. Collect Earnings Call Transcripts

Transcripts are collected for a basket of public companies.

Example tickers:

```text
AAPL
MSFT
NVDA
AMZN
GOOGL
META
TSLA
JPM
