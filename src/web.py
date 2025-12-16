from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
import os
import io
import base64
from pathlib import Path
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg') # Agar tidak error GUI di server
import matplotlib.pyplot as plt

APP = Flask(__name__)

# ==========================================
# 1. KONFIGURASI PATH (DISESUAIKAN DENGAN GAMBAR)
# ==========================================
# Karena web.py ada di dalam 'src', dan model juga di 'src',
# maka path model adalah folder tempat file ini berada.
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR 

print(f"📂 Working Directory: {BASE_DIR}")
print("⏳ Sedang memuat Model BGE-LoRA dari folder yang sama...")

try:
    # Load model langsung dari folder saat ini
    MODEL = SentenceTransformer(str(MODEL_PATH))
    print("✅ Model berhasil dimuat!")
except Exception as e:
    print(f"⚠️ Gagal memuat model.")
    print(f"Error: {e}")
    MODEL = None

# ==========================================
# 2. LOAD DATASET & SPECS
# ==========================================
def load_data():
    """Load data teks diskusi dari reddit_topic.json"""
    json_path = BASE_DIR / 'reddit_topic.json'
    
    texts = []
    if json_path.exists():
        try:
            # Pandas pintar membaca JSON, baik list of dicts maupun structure lain
            df = pd.read_json(json_path)
            
            # Cek kolom yang tersedia
            if 'cleaned_text' in df.columns:
                texts = df['cleaned_text'].tolist()
            elif 'body' in df.columns:
                 texts = df['body'].tolist()
            elif '0' in df.columns: # Kadang json list polos jadi kolom '0'
                 texts = df['0'].astype(str).tolist()
            else:
                # Fallback: ambil kolom pertama apapun namanya
                texts = df.iloc[:, 0].astype(str).tolist()
            print(f"✅ Berhasil memuat {len(texts)} data diskusi Reddit.")
        except Exception as e:
            print(f"⚠️ Error membaca reddit_topic.json: {e}")
    else:
        print("⚠️ File reddit_topic.json tidak ditemukan.")
    
    # Dummy data jika gagal load
    if not texts:
        texts = ["Tesla Model 3 range is good.", "Ioniq 5 charging is fast."] * 10
        
    return texts

def load_car_specs():
    """Load data spesifikasi mobil dari ev_specs.csv"""
    csv_path = BASE_DIR / 'ev_specs.csv'
    
    specs = []
    if csv_path.exists():
        try:
            df_specs = pd.read_csv(csv_path)
            # Konversi ke format list of dict
            for _, row in df_specs.iterrows():
                specs.append({
                    'key': str(row['model_keyword']).lower(),
                    'name': row['real_name'],
                    'price': row['price'],
                    'range': row['range'],
                    'img': row['img_url']
                })
            print(f"✅ Berhasil memuat {len(specs)} data spesifikasi mobil.")
        except Exception as e:
            print(f"⚠️ Error membaca ev_specs.csv: {e}")
    else:
        print("⚠️ File ev_specs.csv tidak ditemukan. Fitur deteksi mobil akan non-aktif.")
            
    return specs

# Load Data saat startup
COMMENTS = load_data()
CAR_SPECS = load_car_specs()

# Pre-compute Embeddings
print("⏳ Memuat database vektor...")

embedding_path = BASE_DIR / 'corpus_embeddings.npy'
need_recalc = False

# 1. Coba Load File NPY
if embedding_path.exists():
    try:
        CORPUS_EMBEDDINGS = np.load(embedding_path)
        print(f"✅ File npy ditemukan. Isi: {len(CORPUS_EMBEDDINGS)} vektor.")
        
        # Cek apakah jumlahnya sinkron dengan Teks JSON
        if len(CORPUS_EMBEDDINGS) != len(COMMENTS):
            print(f"⚠️ MISMATCH: Teks ada {len(COMMENTS)}, tapi Vektor ada {len(CORPUS_EMBEDDINGS)}.")
            print("⚠️ Harus hitung ulang agar sinkron...")
            need_recalc = True
        else:
            print("🚀 Data sinkron! Menggunakan file cache (Cepat).")

    except Exception as e:
        print(f"⚠️ File npy rusak: {e}")
        need_recalc = True
else:
    print("⚠️ File 'corpus_embeddings.npy' tidak ditemukan.")
    need_recalc = True

# 2. Hitung Ulang & SIMPAN (Jika diperlukan)
if need_recalc:
    if MODEL:
        print("⚙️ Menghitung embedding di Laptop (Hanya sekali ini saja)...")
        # Hitung
        CORPUS_EMBEDDINGS = MODEL.encode(COMMENTS, convert_to_numpy=True, show_progress_bar=True)
        
        # --- KUNCI PERBAIKAN: SIMPAN HASILNYA ---
        try:
            np.save(embedding_path, CORPUS_EMBEDDINGS)
            print(f"✅ Berhasil menyimpan {embedding_path}")
            print("🎉 Restart berikutnya pasti cepat!")
        except Exception as e:
            print(f"❌ Gagal menyimpan file npy: {e}")
            
    else:
        CORPUS_EMBEDDINGS = None

print("✅ Siap digunakan!")

# ==========================================
# 3. FUNGSI UTILITY (VISUALISASI)
# ==========================================
def generate_charts(detected_cars, texts):
    # 1. GRAFIK PERBANDINGAN RANGE (KM)
    # Kita ambil data dari mobil yang terdeteksi
    if detected_cars:
        names = [car['name'].split(" ")[0] + " " + car['name'].split(" ")[1] for car in detected_cars] # Ambil 2 kata pertama nama mobil biar gak kepanjangan
        
        # Bersihkan angka range (hapus ' km' dan ubah ke int)
        ranges = []
        for car in detected_cars:
            try:
                r = int(str(car['range']).lower().replace('km', '').strip())
            except:
                r = 0
            ranges.append(r)
        
        # Buat Bar Chart
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        bars = ax1.barh(names, ranges, color='#3b82f6')
        
        # Hiasan Grafik
        ax1.set_xlabel('Jarak Tempuh (km)', fontsize=8)
        ax1.set_title('Perbandingan Baterai', fontsize=10, fontweight='bold')
        ax1.grid(axis='x', linestyle='--', alpha=0.5)
        
        # Tambah label angka di ujung bar
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 10, bar.get_y() + bar.get_height()/2, 
                     f'{int(width)} km', ha='left', va='center', fontsize=8)
            
        plt.tight_layout()
    else:
        # Jika tidak ada mobil terdeteksi, tampilkan placeholder kosong
        fig1, ax1 = plt.subplots(figsize=(5, 1))
        ax1.text(0.5, 0.5, "Data Range Tidak Tersedia", ha='center', va='center')
        ax1.axis('off')

    # Simpan Grafik 1 ke Base64
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight', transparent=True)
    buf1.seek(0)
    chart1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close(fig1)

    # 2. WORD CLOUD (Tetap dipertahankan karena bagus)
    text_combined = " ".join(texts)
    if not text_combined.strip(): text_combined = "No Data"
        
    wordcloud = WordCloud(width=500, height=250, background_color='white', colormap='viridis', max_words=50).generate(text_combined)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis("off")
    ax2.set_title("Topik Diskusi Terkait", fontsize=10)
    
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight', transparent=True)
    buf2.seek(0)
    chart2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close(fig2)
    
    return chart1, chart2

# ==========================================
# 4. ENGINE PENCARIAN (HYBRID)
# ==========================================
def search_engine(query, k=5):
    if MODEL is None: return [], []
    
    # 1. Instruksi BGE (WAJIB UNTUK QUERY)
    instruction = "Represent this sentence for searching relevant passages: "
    query_emb = MODEL.encode(instruction + query, convert_to_numpy=True)
    
    # 2. Hitung Cosine Similarity
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    d_norm = CORPUS_EMBEDDINGS / (np.linalg.norm(CORPUS_EMBEDDINGS, axis=1, keepdims=True) + 1e-12)
    scores = (d_norm @ q_norm)
    
    # 3. Ambil Top K
    top_indices = np.argsort(scores)[::-1][:k]
    
    results = []
    detected_cars = []
    seen_cars = set()
    
    for idx in top_indices:
        text = COMMENTS[idx]
        score = float(scores[idx])
        
        # Analisis Sentimen per item
        sentiment = TextBlob(text).sentiment.polarity
        
        results.append({
            'text': text,
            'score': score,
            'sentiment': sentiment
        })
        
        # Cek apakah teks menyebutkan mobil yang ada di CSV spec kita
        text_lower = text.lower()
        for car in CAR_SPECS:
            if car['key'] in text_lower and car['name'] not in seen_cars:
                detected_cars.append(car)
                seen_cars.add(car['name'])
                
    return results, detected_cars

# ==========================================
# 5. UI HTML (TEMPLATE)
# ==========================================
HTML_TEMPLATE = """
<!doctype html>
<html lang="id">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Sistem Rekomendasi EV</title>
    <style>
      :root { --primary: #0f172a; --accent: #3b82f6; --bg: #f1f5f9; }
      body { font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: #334155; margin: 0; padding: 20px; }
      .container { max-width: 1200px; margin: 0 auto; }
      
      /* Header Simpel */
      .header { text-align: center; margin-bottom: 30px; }
      .header h1 { color: var(--primary); margin: 0; font-size: 2rem; }
      .header span { color: var(--accent); }

      /* Search Bar Menonjol */
      .search-box { background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); display: flex; gap: 10px; margin-bottom: 40px; }
      input { flex: 1; padding: 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 1rem; outline: none; }
      button { background: var(--accent); color: white; border: none; padding: 12px 30px; border-radius: 8px; font-weight: 600; cursor: pointer; }
      button:hover { background: #2563eb; }

      /* === BAGIAN UTAMA: REKOMENDASI MOBIL (Highlight) === */
      .recommendation-section { margin-bottom: 50px; }
      .section-title { font-size: 1.5rem; color: var(--primary); margin-bottom: 20px; border-left: 5px solid var(--accent); padding-left: 15px; }
      
      .car-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px; }
      .car-card { background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); transition: transform 0.2s; border: 1px solid #e2e8f0; }
      .car-card:hover { transform: translateY(-5px); border-color: var(--accent); }
      .car-img-container { height: 200px; overflow: hidden; background: #f8fafc; display: flex; align-items: center; justify-content: center; }
      .car-img { width: 100%; height: 100%; object-fit: cover; }
      .car-details { padding: 20px; }
      .car-name { margin: 0 0 10px; font-size: 1.4rem; color: var(--primary); }
      .car-specs { display: flex; gap: 15px; margin-bottom: 15px; }
      .spec-badge { background: #eff6ff; color: #1e40af; padding: 5px 10px; border-radius: 6px; font-size: 0.9rem; font-weight: 600; }
      
      /* === BAGIAN PENDUKUNG: BUKTI DISKUSI === */
      .evidence-section { display: grid; grid-template-columns: 2fr 1fr; gap: 30px; }
      
      .review-card { background: white; padding: 20px; border-radius: 12px; margin-bottom: 15px; border-left: 4px solid #cbd5e1; }
      .match-score { font-weight: bold; color: var(--accent); }
      
      .chart-card { background: white; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px; }
      .chart-img { max-width: 100%; height: auto; }

      @media (max-width: 768px) { .evidence-section { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h1>Sistem Rekomendasi <span>Kendaraan Listrik</span></h1>
        <p>Berbasis Semantic Search & Transfer Learning</p>
      </div>

      <form method="post" action="/search" class="search-box">
        <input type="text" name="query" placeholder="Contoh: Saya cari mobil keluarga yang nyaman dan baterai awet..." value="{{ query }}">
        <button type="submit">Cari Rekomendasi</button>
      </form>

      {% if results %}
        
        <div class="recommendation-section">
          <h2 class="section-title">🚘 Rekomendasi Produk</h2>
          {% if cars %}
            <div class="car-grid">
              {% for car in cars %}
              <div class="car-card">
                <div class="car-img-container">
                  <img src="{{ car.img }}" class="car-img" alt="{{ car.name }}">
                </div>
                <div class="car-details">
                  <h3 class="car-name">{{ car.name }}</h3>
                  <div class="car-specs">
                    <span class="spec-badge">🔋 {{ car.range }}</span>
                    <span class="spec-badge">💰 {{ car.price }}</span>
                  </div>
                  <p style="font-size: 0.9rem; color: #64748b;">Produk ini cocok dengan kriteria pencarian Anda berdasarkan diskusi komunitas.</p>
                </div>
              </div>
              {% endfor %}
            </div>
          {% else %}
            <div style="background: #fff3cd; padding: 15px; border-radius: 8px;">
              <p>Produk spesifik belum terdeteksi di database spek, namun silakan lihat diskusi di bawah.</p>
            </div>
          {% endif %}
        </div>

        <div class="evidence-section">
          <div>
            <h2 class="section-title">💬 Bukti Diskusi Komunitas</h2>
            <p style="margin-bottom: 20px; opacity: 0.7;">Sistem menemukan diskusi Reddit berikut yang paling relevan dengan kebutuhan Anda:</p>
            {% for res in results %}
            <div class="review-card">
              <p style="font-style: italic; color: #334155;">"{{ res.text[:300] }}..."</p>
              <small>Tingkat Relevansi: <span class="match-score">{{ "%.1f"|format(res.score * 100) }}%</span></small>
            </div>
            {% endfor %}
          </div>

          <div>
            <h2 class="section-title">📊 Analisis Opini</h2>
            <div class="chart-card">
              <h4>Perbandingan Jarak Tempuh</h4>
              <img src="data:image/png;base64,{{ pie_chart }}" class="chart-img" />
              <p style="font-size: 0.8rem; margin-top: 10px; color: #64748b;">Estimasi range (km) berdasarkan spesifikasi model yang direkomendasikan.</p>
            </div>
            <div class="chart-card">
              <h4>Kata Kunci Populer</h4>
              <img src="data:image/png;base64,{{ word_cloud }}" class="chart-img" />
            </div>
          </div>
        </div>

      {% endif %}
    </div>
  </body>
</html>
"""

# ==========================================
# 6. RUN FLASK
# ==========================================
@APP.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, query="")

@APP.route('/search', methods=['POST'])
def search_route():
    query = request.form.get('query', '').strip()
    
    if not query:
        return render_template_string(HTML_TEMPLATE, query="")
        
    # 1. Jalankan Search Engine
    results, cars = search_engine(query, k=5)
    
    # 2. Generate Grafik (PARAMETER BERUBAH DISINI)
    # Kita kirim data 'cars' dan 'texts'
    texts = [r['text'] for r in results]
    
    # Variable namanya kita ganti biar sesuai konteks (range_chart & topic_chart)
    range_chart, topic_chart = generate_charts(cars, texts)
    
    return render_template_string(
        HTML_TEMPLATE, 
        query=query, 
        results=results, 
        cars=cars, 
        # Kirim variabel chart baru ke HTML
        pie_chart=range_chart,  # Kita timpa variabel lama biar gak ubah HTML banyak
        word_cloud=topic_chart
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Server berjalan di http://localhost:{port}")
    APP.run(host='0.0.0.0', port=port, debug=True)