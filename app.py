import yfinance as yf
import openai
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import streamlit as st
from datetime import datetime

# ✅ Streamlit App Setup
st.set_page_config(page_title="Stock Analyzer AI", layout="wide")
st.title("📈 Stock Analyzer with GPT-4")

# ✅ Format numbers with units
def format_currency(n):
    if n is None:
        return "N/A"
    if n >= 1_000_000_000_000:
        return f"${n / 1_000_000_000_000:.2f}T"
    elif n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"${n / 1_000_000:.2f}M"
    else:
        return f"${n:,.0f}"

# ✅ Google News RSS + Article Extraction
# ✅ Google News RSS + Article Extraction (Streamlit Cloud Compatible)
def extract_main_text_from_url(url):
    try:
        page = requests.get(url, timeout=10)
        soup = BeautifulSoup(page.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = '\n'.join(p.get_text() for p in paragraphs)
        return text.strip()[:2000]
    except Exception:
        return ""

def get_article_texts(company_name, count=3):
    search_query = company_name.replace(" ", "+") + "+stock"
    url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"

    articles = []
    try:
        response = requests.get(url)
        root = ET.fromstring(response.content)
        items = root.findall(".//item")
        for item in items[:count]:
            link = item.find("link").text
            title_raw = item.find("title").text.strip()
            description_raw = item.find("description").text.strip() if item.find("description") is not None else ""

            title = BeautifulSoup(title_raw, "html.parser").get_text()
            description = BeautifulSoup(description_raw, "html.parser").get_text()

            main_text = extract_main_text_from_url(link)
            if main_text:
                articles.append(main_text)
            elif description:
                articles.append(description)
            else:
                articles.append(title)
    except Exception as e:
        st.error(f"Error fetching news: {e}")
    return articles
    
# ✅ GPT Setup
openai.api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else ""

def summarize_news(articles):
    combined = "\n\n".join(articles)
    prompt = f"Summarize the likely impact of the following news articles on the company's stock:\n{combined}\n\nAlso label the sentiment of the overall news as one of the following: Positive, Negative, or Mixed. Return the summary first and then the sentiment in a new line like this:\nSentiment: <label>"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600
    )

    content = response['choices'][0]['message']['content']
    sentiment_line = re.search(r"Sentiment:\s*(Positive|Negative|Mixed)", content, re.IGNORECASE)
    if sentiment_line:
        sentiment = sentiment_line.group(1).capitalize()
        emoji = {"Positive": "🟢", "Negative": "🔴", "Mixed": "🟡"}.get(sentiment, "⚪")
        content = re.sub(r"Sentiment:\s*(Positive|Negative|Mixed)", f"Sentiment: {emoji} {sentiment}", content)
    return content

def analyze_financials(info):
    metrics = {
        "P/E Ratio": info.get('trailingPE'),
        "Forward P/E": info.get('forwardPE'),
        "Price to Book": info.get('priceToBook'),
        "ROE": info.get('returnOnEquity'),
        "ROA": info.get('returnOnAssets'),
        "Profit Margin": info.get('profitMargins'),
        "Debt to Equity": info.get('debtToEquity'),
        "Operating Margin": info.get('operatingMargins')
    }
    prompt = f"Analyze the following financial ratios and suggest whether the company appears financially strong or not:\n{metrics}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response['choices'][0]['message']['content']

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info

# ✅ Load list of tickers with company names for suggestions
@st.cache_data
def get_all_tickers():
    response = requests.get("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")
    lines = response.text.strip().split("\n")
    return [(line.split(',')[0], line.split(',')[1]) for line in lines[1:]]

all_companies = get_all_tickers()

# ✅ Autocomplete dropdown using selectbox
company_options = [f"{name} ({ticker})" for ticker, name in all_companies]
selected_company = st.sidebar.selectbox("Search company name", options=company_options)

ticker_input = selected_company.split("(")[-1].strip(")") if selected_company else ""

if ticker_input:
    try:
        info = fetch_stock_data(ticker_input)
        company_name = info.get("longName", ticker_input)

        st.subheader(f"📊 {company_name} ({ticker_input.upper()})")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("P/E Ratio", info.get("trailingPE"))
            st.metric("ROE", info.get("returnOnEquity"))
            st.metric("Operating Margin", info.get("operatingMargins"))

        with col2:
            st.metric("Price/Book", info.get("priceToBook"))
            st.metric("ROA", info.get("returnOnAssets"))
            st.metric("Profit Margin", info.get("profitMargins"))

        with col3:
            st.metric("Debt/Equity", info.get("debtToEquity"))
            st.metric("Revenue (TTM)", format_currency(info.get("totalRevenue")))
            st.metric("Market Cap", format_currency(info.get("marketCap")))

        st.divider()
        st.subheader("📈 Financial Interpretation")
        st.write(analyze_financials(info))

        st.divider()
        st.subheader("📰 Recent News Analysis")
        articles = get_article_texts(company_name)

        with st.expander("View Article Summaries"):
            for i, article in enumerate(articles, 1):
                st.markdown(f"**Article {i}:** {article[:300]}...")

        summary = summarize_news(articles)
        st.markdown("**🧠 GPT Summary:**")
        summary_main = summary.split('\nSentiment:')[0].strip()
        st.success(summary_main)

        sentiment_match = re.search(r"Sentiment:\s*(🟢|🔴|🟡)\s*(Positive|Negative|Mixed)", summary)
        if sentiment_match:
            st.info(f"Sentiment: {sentiment_match.group(1)} {sentiment_match.group(2)}")

    except Exception as e:
        st.error(f"Failed to load data: {e}")
