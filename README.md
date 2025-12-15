# Similarity Dashboard

This simple dashboard loads a local SentenceTransformers model (from the `src` folder where `adapter_model.safetensors` and tokenizer files are located) and lets you input a query and a list of comments (paste or upload a .txt). It returns the top-5 most similar comments.

Quick start:

1. Create a virtual environment (optional):

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or cmd
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the dashboard:

```bash
python src/web.py
```

4. Open http://127.0.0.1:5000 in your browser.

Notes:
- The code expects the model files (including `adapter_model.safetensors`, tokenizer files, and config) to be in the `src` folder next to `web.py`.
- If the model fails to load, the app will show an error message in the results area. Make sure `sentence-transformers` and `safetensors` are installed.
