
# spy_dashboard_ultimate.py

# To run: streamlit run spy_dashboard_ultimate.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import VolumeWeightedAveragePrice
from ta.volatility import AverageTrueRange
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import requests
import datetime
import time
import os
import pickle

nltk.download('vader_lexicon')

# === CONFIG ===
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY_HERE"
SPY_INTERVAL = "5m"
SPY_PERIOD = "3d"
REFRESH_INTERVAL = 60
MODEL_FILE = "clf_model_ultimate.pkl"
DAILY_EXPORT_FILE = "daily_predictions_multiclass.csv"
CONFIDENCE_BASE = 0.6
# ==============

st.set_page_config(page_title="SPY Ultimate Dashboard", layout="wide")
st.title("SPY Ultimate Prediction Dashboard")
st.markdown("Multiclass labels + model stacking + adaptive thresholds + mobile mode.")

@st.cache_data
def fetch_news_sentiment(api_key):
    url = "https://newsapi.org/v2/everything"
    now = datetime.datetime.utcnow()
    from_time = (now - datetime.timedelta(minutes=30)).isoformat()
    params = {
        "q": "SPY OR S&P 500",
        "from": from_time,
        "sortBy": "publishedAt",
        "apiKey": api_key,
        "language": "en",
        "pageSize": 20
    }
    response = requests.get(url, params=params)
    headlines = [article["title"] for article in response.json().get("articles", [])]
    sia = SentimentIntensityAnalyzer()
    if headlines:
        scores = [sia.polarity_scores(h)['compound'] for h in headlines]
        avg_sentiment = sum(scores) / len(scores)
        return avg_sentiment, scores, headlines
    else:
        return 0, [], []

def load_data():
    df = yf.download("SPY", period=SPY_PERIOD, interval=SPY_INTERVAL)
    if df.empty or 'Close' not in df.columns:
        st.error("Error: Could not load SPY data. Please check your internet or try again later.")
        st.stop()
    df['rsi'] = RSIIndicator(close=df['Close']).rsi()
    macd = MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    df['vwap'] = vwap.vwap()
    df['atr'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
    df['price_change'] = df['Close'].diff()
    df['volume_change'] = df['Volume'].pct_change()
    df['return'] = df['Close'].pct_change()
    df['rolling_return_3'] = df['return'].rolling(window=3).sum()
    df['volatility_window'] = df['return'].rolling(window=5).std()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['volatility_regime'] = (df['atr'] > df['atr'].rolling(window=100).quantile(0.75)).astype(int)
    df.dropna(inplace=True)
    return df

df = load_data()
sentiment_score, sentiment_series, headlines = fetch_news_sentiment(NEWSAPI_KEY)
df['sentiment'] = sentiment_score
df['sentiment_change'] = pd.Series(sentiment_series).diff().fillna(0).iloc[-1]

# Label refinement into 5 classes
returns = df['return'].shift(-1)
df['label'] = pd.qcut(returns, q=5, labels=["Strong Down", "Weak Down", "Flat", "Weak Up", "Strong Up"])
df.dropna(inplace=True)

# Feature list
features = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'vwap',
            'atr', 'price_change', 'volume_change', 'sentiment', 'sentiment_change',
            'rolling_return_3', 'volatility_window', 'hour', 'dayofweek', 'volatility_regime']
X = df[features]
le = LabelEncoder()
y = le.fit_transform(df['label'])

# === MODEL STACKING ===
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
else:
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf2 = LogisticRegression(max_iter=200)
    clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    model = VotingClassifier(estimators=[
        ('rf', clf1), ('lr', clf2), ('gb', clf3)
    ], voting='soft')
    model.fit(X, y)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

# Predict latest
latest = X.tail(1)
latest_label = df['label'].tail(1).values[0]
probs = model.predict_proba(latest)[0]
classes = le.classes_
top_class = np.argmax(probs)
confidence = probs[top_class]

# Adaptive confidence: based on rolling accuracy
rolling_predictions = model.predict(X.tail(30))
rolling_true = y[-30:]
rolling_accuracy = accuracy_score(rolling_true, rolling_predictions)
adaptive_threshold = max(CONFIDENCE_BASE, rolling_accuracy)

# Log prediction
timestamp = df.index[-1]
log_entry = pd.DataFrame({
    'timestamp': [timestamp],
    'predicted': [classes[top_class]],
    'actual': [latest_label],
    'confidence': [confidence]
})
if os.path.exists(DAILY_EXPORT_FILE):
    prev = pd.read_csv(DAILY_EXPORT_FILE)
    log_entry = pd.concat([prev, log_entry]).drop_duplicates(subset='timestamp')
log_entry.to_csv(DAILY_EXPORT_FILE, index=False)

# === Streamlit Display ===
st.subheader("Prediction Summary")
st.metric("Predicted Move", f"{classes[top_class]}")
st.metric("Confidence", f"{confidence*100:.2f}%")
st.metric("Adaptive Threshold", f"{adaptive_threshold*100:.2f}%")

if confidence >= adaptive_threshold:
    st.success("High-confidence prediction. Consider acting on this signal.")
else:
    st.warning("Confidence too low — caution advised.")

# News headlines and sentiment
st.subheader("News Sentiment (Last 30 Min)")
st.write(f"Sentiment Score: {sentiment_score:.3f}")
st.write(f"Sentiment Change: {df['sentiment_change'].iloc[-1]:.3f}")
st.markdown("**Recent Headlines:**")
for h in headlines:
    st.markdown(f"- {h}")

# Strategy tester UI
st.sidebar.header("Strategy Tester (Multiclass)")
selected_classes = st.sidebar.multiselect(
    "Select classes to simulate as 'Buy'",
    options=list(classes),
    default=["Strong Up", "Weak Up"]
)
hold = st.sidebar.selectbox("Holding Period (minutes)", [5, 10, 15], index=0)

df['predicted_class'] = model.predict(X)
df['buy_signal'] = df['predicted_class'].apply(lambda x: 1 if le.inverse_transform([x])[0] in selected_classes else 0)
df['entry'] = df['Close']
df['exit'] = df['Close'].shift(-int(hold / 5))
df['pnl'] = (df['exit'] - df['entry']) * df['buy_signal']
df_bt = df[df['buy_signal'] == 1].dropna()
backtest_acc = accuracy_score(df_bt['label'], le.transform(df_bt['predicted_class']))
total_return = df_bt['pnl'].sum()

st.sidebar.markdown(f"**Backtest Accuracy:** {backtest_acc*100:.2f}%")
st.sidebar.markdown(f"**Total Simulated Return:** ${total_return:.2f}")
st.sidebar.download_button("Download Daily Predictions", data=open(DAILY_EXPORT_FILE).read(), file_name="daily_predictions.csv")

# Mobile View
if st.sidebar.checkbox("Mobile Mode"):
    st.metric("Move", f"{classes[top_class]}")
    st.metric("Conf.", f"{confidence*100:.2f}%")
else:
    st.line_chart(df[['Close', 'sma_20', 'sma_50']].tail(100))
