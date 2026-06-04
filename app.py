"""
Stock Market Agent - Web UI
Run with: python app.py
Then open http://localhost:5000
"""

from flask import Flask, render_template_string, request, Response, stream_with_context
from agent import ask_stream, ask
from rag import ingest
import json

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TBD — Market Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #ffffff;
    --surface: #fafafa;
    --border: #e5e5e5;
    --text: #111111;
    --muted: #737373;
    --light-gray: #f4f4f5;
    --user-bg: #f4f4f5;
    --ai-bg: #ffffff;
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0a0a0a;
      --surface: #121212;
      --border: #262626;
      --text: #f5f5f5;
      --muted: #a3a3a3;
      --light-gray: #1a1a1a;
      --user-bg: #1a1a1a;
      --ai-bg: #0a0a0a;
    }
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: background 0.3s, color 0.3s;
  }

  header {
    position: relative;
    z-index: 10;
    padding: 24px 40px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg);
  }

  .brand {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .logo {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.3px;
  }

  .subtitle {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
  }

  .status {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 11px;
    font-weight: 500;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 6px 14px;
    border-radius: 20px;
    background: var(--surface);
  }

  .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--text);
    animation: breath 2.5s ease-in-out infinite;
  }

  @keyframes breath {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
  }

  .chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 40px;
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    gap: 32px;
    max-width: 900px;
    width: 100%;
    margin: 0 auto;
  }

  .chat-area::-webkit-scrollbar { width: 4px; }
  .chat-area::-webkit-scrollbar-thumb { background: var(--border); }

  .message {
    max-width: 720px;
    width: 100%;
    animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  }

  @keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .message.user { margin-left: auto; }

  .message-header {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .message.user .message-header { text-align: right; }

  .bubble {
    padding: 18px 24px;
    border-radius: 12px;
    line-height: 1.65;
    font-size: 15px;
  }

  .message.user .bubble {
    background: var(--user-bg);
    color: var(--text);
    border-bottom-right-radius: 2px;
  }

  .message.ai .bubble {
    background: var(--ai-bg);
    border: 1px solid var(--border);
    color: var(--text);
    border-top-left-radius: 2px;
  }

  .loading-dots {
    display: inline-flex;
    gap: 4px;
    align-items: center;
    padding: 4px 0;
  }

  .loading-dots span {
    width: 4px; height: 4px;
    background-color: var(--text);
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
  }
  .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
  .loading-dots span:nth-child(2) { animation-delay: -0.16s; }

  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1.0); }
  }

  .welcome {
    text-align: center;
    padding: 80px 20px;
    margin: auto 0;
  }

  .welcome h2 {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 16px;
    letter-spacing: -0.5px;
  }

  .welcome h2 span { font-style: italic; font-weight: 400; }

  .welcome p { font-size: 15px; color: var(--muted); max-width: 480px; margin: 0 auto 36px; line-height: 1.6; }

  .welcome .disclaimer {
    font-size: 11px;
    color: var(--muted);
    margin-top: 24px;
    opacity: 0.7;
    max-width: 400px;
    margin-left: auto;
    margin-right: auto;
  }

  .suggestions {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    max-width: 520px;
    margin: 0 auto;
  }

  @media (max-width: 600px) {
    .suggestions { grid-template-columns: 1fr; }
  }

  .suggestion {
    padding: 14px 18px;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    color: var(--text);
    background: var(--surface);
    text-align: left;
    transition: all 0.2s ease;
  }

  .suggestion:hover {
    background: var(--text);
    color: var(--bg);
    border-color: var(--text);
  }

  .input-area {
    position: relative;
    z-index: 10;
    padding: 24px 40px 40px;
    background: var(--bg);
    max-width: 900px;
    width: 100%;
    margin: 0 auto;
  }

  .input-wrapper {
    border: 1px solid var(--border);
    border-radius: 16px;
    background: var(--surface);
    padding: 8px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }

  .input-wrapper:focus-within {
    border-color: var(--text);
  }

  .input-row {
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }

  .ticker-select {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 10px 14px;
    font-size: 12px;
    font-weight: 500;
    border-radius: 10px;
    outline: none;
    cursor: pointer;
    min-width: 110px;
    height: 40px;
  }

  textarea {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text);
    padding: 10px 12px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 400;
    resize: none;
    outline: none;
    min-height: 40px;
    max-height: 120px;
    line-height: 1.5;
  }

  textarea::placeholder { color: var(--muted); }

  .actions-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 8px;
    padding: 0 4px;
  }

  button#send {
    background: var(--text);
    color: var(--bg);
    border: none;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 500;
    border-radius: 10px;
    cursor: pointer;
    transition: opacity 0.2s;
    height: 40px;
  }

  button#send:hover { opacity: 0.9; }
  button#send:disabled { opacity: 0.3; cursor: not-allowed; }

  button#ingest-btn {
    background: transparent;
    color: var(--muted);
    border: none;
    padding: 0 8px;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: color 0.2s;
    height: 40px;
  }

  button#ingest-btn:hover { color: var(--text); }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="logo">TBD</div>
    <div class="subtitle">Market Intelligence</div>
  </div>
  <div class="status">
    <div class="dot"></div>
    Gemma + RAG Engine
  </div>
</header>

<div class="chat-area" id="chat">
  <div class="welcome" id="welcome">
    <h2>Institutional <span>knowledge.</span></h2>
    <p>Analyze equity history, trends, and real-time market reporting through contextualized intelligence.</p>
    
    <div class="suggestions">
      <div class="suggestion" onclick="fillQuery(this)">How has AAPL performed this month?</div>
      <div class="suggestion" onclick="fillQuery(this)">Compare TSLA vs MSFT recently</div>
      <div class="suggestion" onclick="fillQuery(this)">What are the latest GOOGL news?</div>
      <div class="suggestion" onclick="fillQuery(this)">Should I be worried about AMZN?</div>
    </div>

    <p class="disclaimer">
      For informational usage only. Testing systems involve raw asset vulnerabilities. Always consult a certified advisor.
    </p>
  </div>
</div>

<div class="input-area">
  <div class="input-wrapper">
    <div class="input-row">
      <select class="ticker-select" id="ticker-filter">
        <option value="">All tickers</option>
        <option value="AAPL">AAPL</option>
        <option value="MSFT">MSFT</option>
        <option value="GOOGL">GOOGL</option>
        <option value="AMZN">AMZN</option>
        <option value="META">META</option>
        <option value="NVDA">NVDA</option>
        <option value="TSLA">TSLA</option>
        <option value="NFLX">NFLX</option>
        <option value="AMD">AMD</option>
        <option value="INTC">INTC</option>
        <option value="JPM">JPM</option>
        <option value="BAC">BAC</option>
        <option value="GS">GS</option>
        <option value="V">V</option>
        <option value="MA">MA</option>
        <option value="PYPL">PYPL</option>
        <option value="WFC">WFC</option>
        <option value="JNJ">JNJ</option>
        <option value="PFE">PFE</option>
        <option value="MRNA">MRNA</option>
        <option value="UNH">UNH</option>
        <option value="LLY">LLY</option>
        <option value="XOM">XOM</option>
        <option value="CVX">CVX</option>
        <option value="COP">COP</option>
        <option value="BP">BP</option>
        <option value="WMT">WMT</option>
        <option value="TGT">TGT</option>
        <option value="COST">COST</option>
        <option value="NKE">NKE</option>
        <option value="MCD">MCD</option>
        <option value="SBUX">SBUX</option>
        <option value="KO">KO</option>
        <option value="PEP">PEP</option>
        <option value="PG">PG</option>
        <option value="DIS">DIS</option>
        <option value="T">T</option>
        <option value="VZ">VZ</option>
        <option value="SNAP">SNAP</option>
        <option value="SPOT">SPOT</option>
        <option value="F">F</option>
        <option value="GM">GM</option>
        <option value="RIVN">RIVN</option>
        <option value="CRM">CRM</option>
        <option value="ORCL">ORCL</option>
        <option value="SNOW">SNOW</option>
        <option value="PLTR">PLTR</option>
        <option value="UBER">UBER</option>
        <option value="ABNB">ABNB</option>
        <option value="SHOP">SHOP</option>
        <option value="QCOM">QCOM</option>
        <option value="TXN">TXN</option>
        <option value="MU">MU</option>
        <option value="AVGO">AVGO</option>
        <option value="SPY">SPY</option>
        <option value="QQQ">QQQ</option>
        <option value="IWM">IWM</option>
        <option value="VTI">VTI</option>
      </select>
      <textarea id="query" placeholder="Ask about specific listings, portfolios, or news..." rows="1"></textarea>
      <button id="send" onclick="sendMessage()">Send</button>
    </div>
  </div>
  <div class="actions-row">
    <button id="ingest-btn" onclick="runIngest()">Refresh Data Source</button>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const queryEl = document.getElementById('query');
const sendBtn = document.getElementById('send');

queryEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

queryEl.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

function fillQuery(el) {
  queryEl.value = el.textContent;
  queryEl.focus();
}

function addMessage(role, text) {
  document.getElementById('welcome')?.remove();
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;
  const label = role === 'user' ? 'Query' : 'Analysis';
  wrap.innerHTML = `
    <div class="message-header">${label}</div>
    <div class="bubble">${text}</div>
  `;
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap.querySelector('.bubble');
}

async function sendMessage() {
  const query = queryEl.value.trim();
  if (!query) return;
  const ticker = document.getElementById('ticker-filter').value;

  addMessage('user', query);
  queryEl.value = '';
  queryEl.style.height = 'auto';
  sendBtn.disabled = true;

  const aiBubble = addMessage('ai', '');
  const loading = document.createElement('div');
  loading.className = 'loading-dots';
  loading.innerHTML = '<span></span><span></span><span></span>';
  aiBubble.appendChild(loading);

  try {
    const params = new URLSearchParams({ query });
    if (ticker) params.append('ticker', ticker);

    const resp = await fetch(`/ask?${params}`);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let text = '';
    let hasContent = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split('\\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') break;
          try {
            const parsed = JSON.parse(data);
            if (!hasContent) {
              loading.remove();
              hasContent = true;
            }
            text += parsed.text;
            aiBubble.textContent = text;
            chat.scrollTop = chat.scrollHeight;
          } catch {}
        }
      }
    }
  } catch(e) {
    loading.remove();
    aiBubble.textContent = 'Connection error. Check backend or local server nodes.';
  }

  sendBtn.disabled = false;
}

async function runIngest() {
  const btn = document.getElementById('ingest-btn');
  btn.textContent = 'Syncing Records...';
  btn.disabled = true;
  try {
    await fetch('/ingest', { method: 'POST' });
    btn.textContent = 'Database Updated';
    setTimeout(() => { btn.textContent = 'Refresh Data Source'; btn.disabled = false; }, 2000);
  } catch {
    btn.textContent = 'Sync Failed';
    setTimeout(() => { btn.textContent = 'Refresh Data Source'; btn.disabled = false; }, 2000);
  }
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/ask")
def ask_route():
    query = request.args.get("query", "")
    ticker = request.args.get("ticker", None) or None

    def generate():
        for chunk in ask_stream(query, ticker_filter=ticker):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.route("/ingest", methods=["POST"])
def ingest_route():
    ingest()
    return {"status": "ok"}

if __name__ == "__main__":
    print("Starting StockMind Web UI at http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True)