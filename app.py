import streamlit as st
# ✅ Streamlit App Setup
st.set_page_config(page_title="Stock Analyzer AI", layout="wide")
st.title("📈 Stock Analyzer with GPT-4")
import yfinance as yf
from openai import OpenAI
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
from datetime import datetime
import pandas as pd

@st.cache_data(ttl=43200, show_spinner=False)
def analyze_financials_cached(info_json):
    return analyze_financials(info_json)

@st.cache_data(ttl=43200, show_spinner=False)
def summarize_news_cached(articles_list):
    return summarize_news(articles_list)

@st.cache_data(ttl=43200, show_spinner=False)
def summarize_description(desc_text: str):
    prompt = f"Summarize the following company description in about 60 words, keeping full sentences and key points:\n\n{desc_text}"
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

@st.cache_data(ttl=43200)  # cache for 12 hours (43200 seconds)
def get_usd_inr_rate():
    try:
        response = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=INR", timeout=5)
        data = response.json()
        return data["rates"]["INR"]
    except Exception:
        return 83  # fallback

# ✅ Set exchange rate for INR to USD
USD_INR_RATE = get_usd_inr_rate()

openai_client = OpenAI(api_key=st.secrets.get("openai_api_key", ""))

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

def is_relevant_article(article_text, company_name):
    prompt = f"""
You're reviewing a news article mentioning {company_name}:

{article_text}

Decide if this article directly discusses {company_name}'s own business operations, performance, products, leadership, strategy, financials, or major announcements. 
If {company_name} is only *briefly mentioned* in a minor role (e.g., giving a contract or mentioned as a market peer), it's **not** relevant.

Reply only with "Yes" if the article *primarily focuses on* {company_name}. Reply "No" otherwise.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2
        )
        result = response.choices[0].message.content.strip()
        return result.lower().startswith("yes")
    except:
        return False

def get_article_texts(company_name, count=3):
    search_query = company_name.replace(" ", "+") + "+stock"
    url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"

    articles = []
    try:
        response = requests.get(url)
        root = ET.fromstring(response.content)
        items = root.findall(".//item")

        for item in items:
            if len(articles) >= count:
                break

            link = item.find("link").text
            title_raw = item.find("title").text.strip()
            description_raw = item.find("description").text.strip() if item.find("description") is not None else ""

            title = BeautifulSoup(title_raw, "html.parser").get_text()
            description = BeautifulSoup(description_raw, "html.parser").get_text()
            main_text = extract_main_text_from_url(link)

            content = main_text if main_text else (description if description else title)

            if content and is_relevant_article(content, company_name):
                articles.append(content)

    except Exception as e:
        st.error(f"Error fetching news: {e}")
    return articles

def get_market_summary(company_exchange, count: int = 1):
    """
    Fetches the top ‘market closed’ summary articles for a given exchange.
    """
    # 1. pick a site-specific query
    if company_exchange == "NSE":
        query = "site:economictimes.indiatimes.com market closed summary India"
    else:
        # generic US-market closed summary – you can refine sites
        query = f"site:moneycontrol.com market closed summary {company_exchange}"
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    )

    summaries = []
    try:
        resp = requests.get(rss_url, timeout=5)
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:count]

        for item in items:
            link = item.find("link").text
            # try to pull some main text, fallback to title/description
            text = extract_main_text_from_url(link) or (
                BeautifulSoup(item.find("title").text, "html.parser").get_text()
            )
            summaries.append(text)
    except Exception as e:
        st.error(f"Market‐summary fetch error: {e}")

    return summaries

def summarize_news(articles):
    combined = "\n\n".join(articles)
    prompt = f"""
You are a stock analyst assistant. Read the following articles related to a specific company and write a summary highlighting key developments that directly impact the company's stock.

Be specific. Include:
- any earnings or revenue numbers mentioned
- product launches or strategic decisions
- acquisitions, partnerships, or leadership changes
- any mentioned stock price movement
- recommendation to buy or sell the stock
- Take geo-political tensions into account
- Avoid generic summaries

Articles:
{combined}

Now write a clear, specific summary of the likely impact on the company's stock with recommendation to buy or sell, followed by a sentiment tag: Positive, Negative, or Mixed.
Format:
Summary: <summary text>
Sentiment: <label>
"""

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600
    )

    content = response.choices[0].message.content.strip()
    sentiment_line = re.search(r"Sentiment:\s*(Positive|Negative|Mixed)", content, re.IGNORECASE)
    if sentiment_line:
        sentiment = sentiment_line.group(1).capitalize()
        emoji = {"Positive": "🟢", "Negative": "🔴", "Mixed": "🟡"}.get(sentiment, "⚪")
        content = re.sub(r"Sentiment:\s*(Positive|Negative|Mixed)", f"Sentiment: {emoji} {sentiment}", content)
    return content

def safe_round(value):
    return round(value, 2) if value is not None else "N/A"

def analyze_financials(info):
    metrics = {
        "P/E Ratio": info.get('trailingPE'),
        "Forward P/E": info.get('forwardPE'),
        "Price to Book": info.get('priceToBook'),
        "ROE": safe_round(info.get('returnOnEquity') * 100) if info.get('returnOnEquity') is not None else "N/A",
        "ROA": safe_round(info.get('returnOnAssets') * 100) if info.get('returnOnAssets') is not None else "N/A",
        "Profit Margin": safe_round(info.get('profitMargins') * 100) if info.get('profitMargins') is not None else "N/A",
        "Debt to Equity": round(info.get('debtToEquity') / 100, 2) if info.get('debtToEquity') is not None else "N/A",
        "Operating Margin": safe_round(info.get('operatingMargins') * 100) if info.get('operatingMargins') is not None else "N/A"
    }

    prompt = f"Analyze the following financial ratios and suggest whether the company appears financially strong or not:\n{metrics}"

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600
    )

    return response.choices[0].message.content

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info

# ✅ Load list of tickers with company names for suggestions
@st.cache_data
def get_all_tickers():
    url = "https://raw.githubusercontent.com/shitij231096/stock-analyzer-ai/main/All_Stock_Listings.csv"
    df = pd.read_csv(url)
    df.columns = [col.strip() for col in df.columns]  # Clean up headers
    return df[['Symbol', 'Company Name', 'Exchange']]

all_companies = get_all_tickers()

# ✅ Sidebar Info
st.sidebar.markdown("🔎 *Currently supports NSE, NYSE, and NASDAQ stocks only.*")

# ✅ Exchange Filter UI
exchange_filter = st.sidebar.selectbox("Select Exchange", options=["NSE", "NYSE", "NASDAQ"])

# ✅ Filter DataFrame Based on Selection
filtered_companies = all_companies if exchange_filter == "All" else all_companies[all_companies["Exchange"] == exchange_filter]

# ✅ Dropdown Options
company_options = [f"{row['Company Name']} ({row['Symbol']})" for _, row in filtered_companies.iterrows()]
selected_company = st.sidebar.selectbox("Search company name", options=company_options)

ticker_input = selected_company.split("(")[-1].strip(")") if selected_company else ""

if ticker_input:
    try:
        exchange_name = all_companies[all_companies['Symbol'] == ticker_input]['Exchange'].values[0]
        info = fetch_stock_data(ticker_input)
        company_name = info.get("longName", ticker_input)
        
        st.subheader(f"📊 {company_name} ({ticker_input.upper()}) - {exchange_name}")
        st.caption(f"Exchange: {exchange_name} | Ticker: {ticker_input.upper()}")
        
        # Show a GPT-generated ~60-word company description
        description = info.get("longBusinessSummary", "")
        if description:
            with st.spinner("Summarizing company profile…"):
                brief_desc = summarize_description(description)
            st.write(brief_desc)

        
        col1, col2, col3 = st.columns(3)

        with col1:
            price = info.get("regularMarketPrice")
            prev_close = info.get("previousClose")

            if price is not None and prev_close:
                change = price - prev_close
                pct_change = (change / prev_close) * 100
                currency_symbol = "₹" if exchange_name == "NSE" else "$"
                st.metric("Last Price", f"{currency_symbol}{price:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
            else:
                st.metric("Last Price", "N/A")
                
            st.metric("P/E Ratio", f'{info.get("trailingPE"):.2f}' if info.get("trailingPE") is not None else "N/A")
            st.metric("ROE", f'{info.get("returnOnEquity") * 100:.2f}%' if info.get("returnOnEquity") is not None else "N/A")
            st.metric("Operating Margin", f'{info.get("operatingMargins") * 100:.2f}%' if info.get("operatingMargins") is not None else "N/A")

        with col2:
            st.metric("Price/Book", f'{info.get("priceToBook"):.2f}' if info.get("priceToBook") is not None else "N/A")
            st.metric("ROA", f'{info.get("returnOnAssets") * 100:.2f}%' if info.get("returnOnAssets") is not None else "N/A")
            st.metric("Profit Margin", f'{info.get("profitMargins") * 100:.2f}%' if info.get("profitMargins") is not None else "N/A")

        with col3:
            st.metric("Debt/Equity", f'{info.get("debtToEquity") / 100:.2f}' if info.get("debtToEquity") is not None else "N/A")
            if exchange_name == "NSE":
                revenue_usd = info.get("totalRevenue") / USD_INR_RATE if info.get("totalRevenue") else None
                market_cap_usd = info.get("marketCap") / USD_INR_RATE if info.get("marketCap") else None
            else:
                revenue_usd = info.get("totalRevenue")
                market_cap_usd = info.get("marketCap")

            st.metric("Revenue (TTM)", format_currency(revenue_usd))
            st.metric("Market Cap", format_currency(market_cap_usd))

        
        st.divider()
        st.subheader("📈 Financial Interpretation")
        with st.spinner("Analyzing financial metrics…"):
            interpretation = analyze_financials_cached(info)
        st.write(interpretation)

        st.divider()
        st.subheader("📰 Recent News Analysis")
        # 1) Company‐specific articles
        articles = get_article_texts(company_name)

        # 2) One extra “market closed” summary
        market_articles = get_market_summary(exchange_name, count=1)
        if market_articles:
            articles.extend(market_articles)
        
        with st.expander("View Article Summaries"):
            for i, article in enumerate(articles, 1):
                st.markdown(f"**Article {i}:** {article[:300]}...")

        if articles:
            with st.expander("📰 View Raw Articles Pulled", expanded=True):
                for i, article in enumerate(articles, 1):
                    st.markdown(f"**Article {i}:**\n\n{article[:800]}...\n\n---")
        else:
            st.info("No relevant articles were found.")
        
        if articles:
            with st.spinner("Summarizing recent news…"):
                summary = summarize_news_cached(articles)
            st.markdown("**🧠 AI Summary:**")
            
            summary_main = summary.split('\nSentiment:')[0].strip()
            st.success(summary_main)

            sentiment_match = re.search(r"Sentiment:\s*(🟢|🔴|🟡)\s*(Positive|Negative|Mixed)", summary)
            if sentiment_match:
                st.info(f"Sentiment: {sentiment_match.group(1)} {sentiment_match.group(2)}")
        else:
            st.warning("No summary generated as no relevant articles were found.")

    except Exception as e:
        st.error(f"Failed to load data: {e}")
