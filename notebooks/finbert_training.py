# ============================================================
# Earnings Call Sentiment Analysis with FinBERT
# ============================================================

# Install dependencies first:
# pip install transformers torch pandas numpy yfinance matplotlib scikit-learn

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from transformers import pipeline
from datetime import datetime, timedelta

# ============================================================
# STEP 1 — LOAD FINBERT MODEL
# ============================================================

print("Loading FinBERT model...")

classifier = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

print("FinBERT loaded successfully.")

# ============================================================
# STEP 2 — SAMPLE EARNINGS CALL TRANSCRIPTS
# ============================================================

# Example transcripts
# Replace these with real earnings call transcripts later

transcripts = [
    {
        "ticker": "AAPL",
        "date": "2024-01-25",
        "text": """
        We delivered record quarterly revenue driven by strong iPhone demand.
        Services revenue reached an all-time high and margins improved significantly.
        We remain optimistic about future growth opportunities.
        """
    },
    {
        "ticker": "TSLA",
        "date": "2024-01-24",
        "text": """
        Vehicle production faced temporary headwinds during the quarter.
        Margins declined due to pricing pressure and macroeconomic uncertainty.
        However, we expect operational improvements going forward.
        """
    },
    {
        "ticker": "NVDA",
        "date": "2024-02-21",
        "text": """
        Demand for AI infrastructure remains exceptionally strong.
        Data center revenue exceeded expectations and guidance was raised substantially.
        """
    }
]

transcripts_df = pd.DataFrame(transcripts)

# ============================================================
# STEP 3 — RUN FINBERT SENTIMENT ANALYSIS
# ============================================================

def get_finbert_sentiment(text):
    
    result = classifier(text[:512])[0]
    
    label = result['label']
    score = result['score']
    
    if label == "positive":
        sentiment_score = score
        
    elif label == "negative":
        sentiment_score = -score
        
    else:
        sentiment_score = 0
        
    return pd.Series([label, score, sentiment_score])

transcripts_df[
    ["finbert_label", "confidence", "sentiment_score"]
] = transcripts_df["text"].apply(get_finbert_sentiment)

print("\nFinBERT Sentiment Results:")
print(transcripts_df[
    ["ticker", "finbert_label", "confidence", "sentiment_score"]
])

# ============================================================
# STEP 4 — DOWNLOAD STOCK PRICE DATA
# ============================================================

def get_forward_returns(ticker, earnings_date):
    
    earnings_date = pd.to_datetime(earnings_date)
    
    start = earnings_date
    end = earnings_date + timedelta(days=10)
    
    stock = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False
    )
    
    if len(stock) < 6:
        return pd.Series([np.nan]*5)
    
    prices = stock["Adj Close"]
    
    returns = []
    
    for i in range(1, 6):
        
        try:
            r = (
                prices.iloc[i] - prices.iloc[0]
            ) / prices.iloc[0]
            
            returns.append(r)
            
        except:
            returns.append(np.nan)
    
    return pd.Series(returns)

transcripts_df[
    ["return_1d", "return_2d", "return_3d", "return_4d", "return_5d"]
] = transcripts_df.apply(
    lambda row: get_forward_returns(
        row["ticker"],
        row["date"]
    ),
    axis=1
)

print("\nForward Returns:")
print(transcripts_df[
    [
        "ticker",
        "return_1d",
        "return_3d",
        "return_5d"
    ]
])

# ============================================================
# STEP 5 — EVENT STUDY ANALYSIS
# ============================================================

correlation_1d = transcripts_df[
    ["sentiment_score", "return_1d"]
].corr().iloc[0,1]

correlation_5d = transcripts_df[
    ["sentiment_score", "return_5d"]
].corr().iloc[0,1]

print("\nEvent Study Results")
print("=" * 40)

print(f"Correlation with 1-Day Returns: {correlation_1d:.4f}")
print(f"Correlation with 5-Day Returns: {correlation_5d:.4f}")

# ============================================================
# STEP 6 — LONG / SHORT STRATEGY
# ============================================================

# Long positive sentiment
# Short negative sentiment

transcripts_df["strategy_return"] = np.where(
    transcripts_df["sentiment_score"] > 0,
    transcripts_df["return_5d"],
    -transcripts_df["return_5d"]
)

average_strategy_return = transcripts_df[
    "strategy_return"
].mean()

print("\nStrategy Performance")
print("=" * 40)

print(
    f"Average Long/Short Return: "
    f"{average_strategy_return:.4%}"
)

# ============================================================
# STEP 7 — PLOT RESULTS
# ============================================================

plt.figure(figsize=(8,5))

plt.scatter(
    transcripts_df["sentiment_score"],
    transcripts_df["return_5d"]
)

plt.xlabel("FinBERT Sentiment Score")
plt.ylabel("5-Day Forward Return")
plt.title("Sentiment vs Future Returns")

plt.grid(True)

plt.show()

# ============================================================
# STEP 8 — FINAL OUTPUT
# ============================================================

print("\nFinal Dataset")
print("=" * 40)

print(transcripts_df)

# ============================================================
# PROJECT COMPLETE
# ============================================================

print("\nProject completed successfully.")
