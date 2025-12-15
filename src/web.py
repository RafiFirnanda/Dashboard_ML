from flask import Flask, request, redirect
from flask import render_template_string
from pathlib import Path
import numpy as np
import os

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
  from transformers import AutoTokenizer
except Exception:
  AutoTokenizer = None

APP = Flask(__name__)

MODEL_DIR = Path(__file__).parent

def load_model():
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. See requirements.txt")
    # Attempt to load model from the current folder (where adapter_model.safetensors exists)
    try:
        model = SentenceTransformer(str(MODEL_DIR))
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load SentenceTransformer from {MODEL_DIR}: {e}")

MODEL = None
TOKENIZER = None

def get_tokenizer():
  global TOKENIZER
  if TOKENIZER is not None:
    return TOKENIZER
  if AutoTokenizer is None:
    return None
  try:
    TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    return TOKENIZER
  except Exception:
    # fall back to None if tokenizer can't be loaded
    TOKENIZER = None
    return None


def tokenize_texts(texts, max_length=512):
  """Return tokenization dict (input_ids, attention_mask) for a list of texts.
  Returns None if tokenizer not available.
  """
  tok = get_tokenizer()
  if tok is None:
    return None
  if isinstance(texts, str):
    texts = [texts]
  return tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL

def embed_texts(texts):
    model = get_model()
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return emb

def top_k_similar(query, docs, k=5):
    if not docs:
        return []
    texts = [query] + docs
    embeddings = embed_texts(texts)
    q = embeddings[0]
    ds = embeddings[1:]
    # cosine similarity
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    ds_norm = ds / (np.linalg.norm(ds, axis=1, keepdims=True) + 1e-12)
    sims = (ds_norm @ q_norm).tolist()
    idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    return [(docs[i], sims[i]) for i in idxs]


def _collect_strings(obj, out):
  # recursively collect string values from nested JSON structures
  if obj is None:
    return
  if isinstance(obj, str):
    s = obj.strip()
    if s:
      out.append(s)
    return
  if isinstance(obj, dict):
    for v in obj.values():
      _collect_strings(v, out)
    return
  if isinstance(obj, list):
    for v in obj:
      _collect_strings(v, out)
    return


def load_default_comments(max_items=1000):
    """Load reddit_topic.json from the same folder as this script and return list of strings.
    The user will place `reddit_topic.json` into the `src/` folder next to `web.py`.
    """
    p = Path(__file__).parent / 'reddit_topic.json'
    try:
        if not p.exists():
            return []
        import json
        with open(p, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        out = []
        _collect_strings(data, out)
        # filter short items and dedupe while preserving order
        seen = set()
        cleaned = []
        for s in out:
            if len(s) < 20:
                continue
            if s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
            if len(cleaned) >= max_items:
                break
        return cleaned
    except Exception:
        return []

HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Similarity Dashboard</title>
    <style>
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 2rem 1rem;
        position: relative;
        overflow-x: hidden;
      }
      body::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="20" cy="20" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="60" cy="60" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="80" cy="20" r="2" fill="rgba(255,255,255,0.1)"/></svg>');
        pointer-events: none;
        z-index: 0;
      }
      .container {
        max-width: 900px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.25), 0 0 0 1px rgba(255,255,255,0.5) inset;
        padding: 2.5rem;
        backdrop-filter: blur(10px);
        position: relative;
        z-index: 1;
      }
      h1 {
        color: #333;
        margin-bottom: 0.5rem;
        font-size: 2.2rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        letter-spacing: -0.5px;
      }
      .subtitle {
        color: #666;
        margin-bottom: 2rem;
        font-size: 0.95rem;
        line-height: 1.4;
      }
      .form-group {
        margin-bottom: 1.5rem;
      }
      label {
        display: block;
        color: #333;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 1rem;
      }
      input[type="text"],
      textarea,
      input[type="file"] {
        width: 100%;
        padding: 0.85rem;
        border: 2px solid rgba(230, 230, 250, 0.6);
        border-radius: 8px;
        font-family: inherit;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: rgba(248, 248, 255, 0.7);
      }
      input[type="text"]:focus,
      textarea:focus {
        outline: none;
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15), inset 0 0 10px rgba(102, 126, 234, 0.05);
        background: rgba(255, 255, 255, 0.9);
      }
      textarea {
        resize: vertical;
        height: 150px;
        min-height: 120px;
      }
      input[type="file"] {
        padding: 0.5rem;
      }
      .help-text {
        color: #666;
        font-size: 0.85rem;
        margin-top: 0.3rem;
      }
      button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.95rem 2.5rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        letter-spacing: 0.5px;
      }
      button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
      }
      button:active {
        transform: translateY(-1px);
      }
      .results-section {
        margin-top: 2.5rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(230, 230, 250, 0.6);
      }
      .results-title {
        color: #333;
        font-size: 1.6rem;
        margin-bottom: 1.8rem;
        display: flex;
        align-items: center;
        font-weight: 600;
      }
      .results-title::before {
        content: "✓";
        display: inline-block;
        margin-right: 0.75rem;
        color: #667eea;
        font-weight: bold;
        font-size: 1.8rem;
      }
      .results-list {
        list-style: none;
      }
      .result-item {
        background: linear-gradient(135deg, rgba(248, 249, 250, 0.9) 0%, rgba(240, 242, 255, 0.7) 100%);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(230, 230, 250, 0.5);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
      }
      .result-item:hover {
        background: linear-gradient(135deg, rgba(240, 242, 255, 1) 0%, rgba(230, 235, 255, 0.9) 100%);
        border-left-color: #764ba2;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        transform: translateX(4px);
      }
      .result-text {
        color: #333;
        margin-bottom: 0.75rem;
        line-height: 1.5;
        font-size: 0.95rem;
      }
      .result-score {
        display: flex;
        align-items: center;
        gap: 0.5rem;
      }
      .score-label {
        color: #666;
        font-size: 0.85rem;
        font-weight: 600;
      }
      .score-value {
        color: #667eea;
        font-weight: bold;
        font-size: 1rem;
      }
      .score-bar {
        width: 100px;
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        overflow: hidden;
        margin-left: 0.5rem;
      }
      .score-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.3s;
      }
      .empty-message {
        text-align: center;
        color: #999;
        padding: 2rem;
        font-size: 0.95rem;
      }
      .error-message {
        background: #fff3cd;
        color: #856404;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 6px;
        margin-top: 1rem;
      }
      .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        text-align: center;
        line-height: 30px;
        font-weight: bold;
        margin-right: 0.75rem;
        font-size: 0.9rem;
      }
      @media (max-width: 600px) {
        .container {
          padding: 1.5rem;
        }
        h1 {
          font-size: 1.5rem;
        }
        .result-item {
          padding: 1rem;
        }
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>⚡ Electric Vehicle Similarity Dashboard</h1>
      <p class="subtitle">Find 5 electric car reviews/data that most closely match your criteria.</p>
      
      <form method="post" action="/search">
        <div class="form-group">
          <label for="query">Criteria for the Electric Car You Want</label>
          <input type="text" id="query" name="query" placeholder="Examples: 'long range battery performance' or 'affordable energy efficient'..." required>
        </div>

        <button type="submit">🔍 Find Most 5 Car</button>
      </form>

      {% if results is defined %}
        <div class="results-section">
          {% if query_text %}
            <div style="background:#f0f2ff; padding:1rem; border-radius:6px; margin-bottom:1.5rem; border-left:4px solid #667eea;">
              <strong style="color:#333;">Criteria that you looking for:</strong>
              <p style="color:#555; margin-top:0.5rem; font-size:0.95rem;">{{ query_text|e }}</p>
            </div>
          {% endif %}
          <h2 class="results-title">Result Electric Car</h2>
          {% if results %}
            <ol class="results-list">
            {% for text,score in results %}
              <li class="result-item">
                <div style="display: flex; gap: 1rem;">
                  <span class="rank-badge">{{ loop.index }}</span>
                  <div style="flex: 1;">
                    <div class="result-text">{{ text|e }}</div>
                    <div class="result-score">
                      <span class="score-label">Kesamaan:</span>
                      <span class="score-value">{{ "%.1f"|format(score * 100) }}%</span>
                      <div class="score-bar">
                        <div class="score-fill" style="width: {{ score * 100 }}%"></div>
                      </div>
                    </div>
                  </div>
                </div>
              </li>
            {% endfor %}
            </ol>
          {% else %}
            <div class="empty-message">
              ❌ Tidak ada data untuk dibandingkan. Silakan masukkan beberapa data/review mobil listrik terlebih dahulu.
            </div>
          {% endif %}
        </div>
      {% endif %}
    </div>
  </body>
</html>
"""


@APP.route('/', methods=['GET'])
def index():
    return render_template_string(HTML)


@APP.route('/search', methods=['POST'])
def search():
  query = (request.form.get('query') or '').strip()
  # comments are not provided via UI anymore; we rely on src/reddit_topic.json as default dataset
  comments = []

  used_default = False
  # If user did not provide comments, try to load default dataset from reddit_topic.json
  if not comments:
    default_comments = load_default_comments(max_items=2000)
    if default_comments:
      comments = default_comments
      used_default = True

  results = []
  if query:
    try:
      results = top_k_similar(query, comments, k=5)
    except Exception as e:
      # on model load error, show message as a single result
      results = [(f"Error: {e}", 0.0)]

  default_count = len(comments) if used_default else 0
  return render_template_string(HTML, results=results, used_default=used_default, default_count=default_count, query_text=query)


if __name__ == '__main__':
    # run development server
    port = int(os.environ.get('PORT', 5000))
    APP.run(host='0.0.0.0', port=port, debug=True)
