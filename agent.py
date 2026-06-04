"""
Stock Market Agent
Connects RAG (ChromaDB + LM Studio embeddings) with Gemma 2B Instruct.
"""

import requests
import json
from rag import retrieve

LM_STUDIO_URL = "http://127.0.0.1:1234"
LM_MODEL = "gemma-2-2b-it"

SYSTEM_PROMPT = """You are a precise, data-driven financial intelligence engine. Analyze the provided Market Data to answer the user's Question.

CRITICAL FORMATTING BOUNDS:
- Never use markdown styling, lists, asterisks (**), or bullet points. Output clean, raw prose.
- Absolutely NO generic disclaimer templates ("I am not a financial advisor", "Investing involves risk"). Provide only factual evaluations based on your database context.

Execute the response strictly according to the detected intent below:

1. INTENT: Identity Lookup (Contains "name", "ticker", "stand for", "symbol")
- Action: Identify the company from the data and print its absolute full legal name followed by its symbol. Do not just return the ticker back to the user.
- Blueprint: [Full Legal Company Name] ([Ticker])

2. INTENT: Profile / Operations (Contains "what does it do", "business model", "sector")
- Action: Synthesize a definitive statement describing their corporate function and market activities.
- Blueprint: [Company Name] operates as a major corporation in the market. Key Operations: [Synthesized description of their primary commercial operations].

3. INTENT: Investment Query / Threat Check (Contains "should I invest", "buy", "sell")
- Action: Scan context for pricing anomalies or multi-day drops. If a severe price contraction is found, warn immediately.
- Blueprint: Trend analysis indicates [Direction]. Volatility Warning: [Explicit data regarding severe price drops if detected in the logs].

4. INTENT: Deep-Dive / Historical Analysis (Contains "analysis", "deeper", "timeline", "metrics", "insides", "history")
- Action: OVERRIDE CONCISE MODES. Extract every single numerical data point, historical date, trading volume shift, or news narrative present in the retrieved context. Synthesize an extensive chronological review tracking their specific daily metrics, pricing variances, and institutional movements. Be thorough."""


def ask(query: str, ticker_filter: str = None) -> str:
    context = retrieve(query, n_results=5, ticker_filter=ticker_filter)

    prompt = f"""Market data:
{context}

Question: {query}

Apply the exact intent pattern rules from your instructions."""

    # Dynamic token allocation based on query type to prevent truncated outputs
    is_deep_dive = any(w in query.lower() for w in ["deeper", "analysis", "timeline", "insides", "history", "metrics"])
    token_limit = 450 if is_deep_dive else 150

    response = requests.post(
        f"{LM_STUDIO_URL}/v1/chat/completions",
        json={
            "model": LM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # Kept minimal to force exact pattern reproduction
            "max_tokens": token_limit,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def ask_stream(query: str, ticker_filter: str = None):
    context = retrieve(query, n_results=5, ticker_filter=ticker_filter)

    prompt = f"""Market data:
{context}

Question: {query}

Apply the exact intent pattern rules from your instructions."""

    is_deep_dive = any(w in query.lower() for w in ["deeper", "analysis", "timeline", "insides", "history", "metrics"])
    token_limit = 450 if is_deep_dive else 150

    response = requests.post(
        f"{LM_STUDIO_URL}/v1/chat/completions",
        json={
            "model": LM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": token_limit,
            "stream": True
        },
        stream=True
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue


if __name__ == "__main__":
    print("Testing agent...")
    result = ask("What is the full name of PYPL?")
    print(result)