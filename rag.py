"""
RAG Component - Stock Market Agent
Fetches live financial news + stock data via yfinance,
embeds them, stores in ChromaDB, and retrieves relevant context.
"""

import yfinance as yf
import pandas as pd
import chromadb
from chromadb import EmbeddingFunction
from datetime import datetime
import hashlib
import requests

# ─── CONFIG ───────────────────────────────────────────────────────────────────

TICKERS = [
    # Big Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "INTC",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA", "PYPL", "WFC", "C",
    # Healthcare
    "JNJ", "PFE", "MRNA", "UNH", "ABBV", "LLY", "MRK", "ABT",
    # Energy
    "XOM", "CVX", "COP", "BP", "OXY",
    # Consumer
    "WMT", "TGT", "COST", "NKE", "MCD", "SBUX", "KO", "PEP", "PG", "DIS",
    # Telecom & Media
    "T", "VZ", "CMCSA", "TMUS", "SNAP", "SPOT",
    # EVs & Automotive
    "F", "GM", "RIVN",
    # Cloud & SaaS
    "CRM", "ORCL", "SNOW", "PLTR", "UBER", "ABNB", "SHOP",
    # Semiconductors
    "QCOM", "TXN", "MU", "AVGO", "AMAT",
    # ETFs
    "SPY", "QQQ", "IWM", "VTI",
]
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "stock_rag"

# Must match exactly what LM Studio shows as "API Model Identifier"
LM_STUDIO_EMBEDDING_MODEL = "text-embedding-embeddinggemma-300m"
LM_STUDIO_URL = "http://127.0.0.1:1234"


# ─── EMBEDDING FUNCTION ───────────────────────────────────────────────────────

class LMStudioEmbedding(EmbeddingFunction):
    def __call__(self, texts):
        response = requests.post(
            f"{LM_STUDIO_URL}/v1/embeddings",
            json={"input": texts, "model": LM_STUDIO_EMBEDDING_MODEL}
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]


# ─── SETUP CHROMADB ───────────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = LMStudioEmbedding()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ─── DATA FETCHING ────────────────────────────────────────────────────────────

def fetch_stock_summary(ticker: str) -> list[dict]:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")
    chunks = []

    for date, row in hist.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        change = row['Close'] - row['Open']
        direction = "up" if change >= 0 else "down"
        pct = abs(change / row['Open'] * 100)

        text = (
            f"On {date_str}, {ticker} opened at ${row['Open']:.2f} and closed at "
            f"${row['Close']:.2f}, moving {direction} by {pct:.2f}%. "
            f"The intraday high was ${row['High']:.2f} and low was ${row['Low']:.2f}. "
            f"Trading volume was {int(row['Volume']):,} shares."
        )
        chunks.append({
            "id": hashlib.md5(f"{ticker}_{date_str}".encode()).hexdigest(),
            "text": text,
            "metadata": {"ticker": ticker, "date": date_str, "type": "price"}
        })

    return chunks


def fetch_news(ticker: str) -> list[dict]:
    stock = yf.Ticker(ticker)
    chunks = []

    try:
        news_items = stock.news or []
        for item in news_items[:10]:
            title = item.get("title", "")
            summary = item.get("summary", "")
            pub_date = datetime.fromtimestamp(
                item.get("providerPublishTime", 0)
            ).strftime("%Y-%m-%d")

            if not title:
                continue

            text = f"News about {ticker} on {pub_date}: {title}."
            if summary:
                text += f" {summary}"

            chunks.append({
                "id": hashlib.md5(f"{ticker}_news_{title}".encode()).hexdigest(),
                "text": text,
                "metadata": {
                    "ticker": ticker,
                    "date": pub_date,
                    "type": "news",
                    "source": item.get("publisher", "unknown")
                }
            })
    except Exception as e:
        print(f"[WARNING] Could not fetch news for {ticker}: {e}")

    return chunks


# ─── INGESTION ────────────────────────────────────────────────────────────────

def ingest(tickers: list[str] = TICKERS):
    collection = get_collection()
    all_chunks = []

    for ticker in tickers:
        print(f"[INFO] Fetching data for {ticker}...")
        all_chunks.extend(fetch_stock_summary(ticker))
        all_chunks.extend(fetch_news(ticker))

    if not all_chunks:
        print("[WARNING] No data fetched.")
        return

    collection.upsert(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks]
    )
    print(f"[INFO] Ingested {len(all_chunks)} chunks into ChromaDB.")


# ─── RETRIEVAL ────────────────────────────────────────────────────────────────

def retrieve(query: str, n_results: int = 5, ticker_filter: str = None) -> str:
    collection = get_collection()
    where = {"ticker": ticker_filter} if ticker_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return "No relevant stock information found."

    context_parts = []
    for doc, meta in zip(docs, metas):
        source = f"[{meta['ticker']} | {meta['date']} | {meta['type']}]"
        context_parts.append(f"{source}\n{doc}")

    return "\n\n".join(context_parts)


# ─── QUICK TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verify LM Studio is reachable before doing anything
    try:
        test = requests.get(f"{LM_STUDIO_URL}/v1/models")
        test.raise_for_status()
        print("[INFO] LM Studio server is reachable.")
    except Exception as e:
        print(f"[ERROR] Cannot reach LM Studio at {LM_STUDIO_URL}: {e}")
        print("Make sure LM Studio is running with the embedding model loaded.")
        exit(1)

    print("\n=== Ingesting stock data ===")
    ingest(["AAPL", "TSLA"])

    print("\n=== Test retrieval ===")
    query = "How has Apple stock performed recently?"
    context = retrieve(query, n_results=3)
    print(f"Query: {query}\n")
    print("Retrieved context:")
    print(context)
