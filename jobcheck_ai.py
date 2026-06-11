# CLS POOLING
import streamlit as st
import torch
import pandas as pd
import numpy as np
import os
import re 
import gspread
from io import BytesIO
from datetime import datetime
from difflib import SequenceMatcher

# Menghilangkan log peringatan TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from transformers import AutoTokenizer, AutoModel
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.models import load_model as load_keras_model

# Konfigurasi Halaman
st.set_page_config(page_title="JobCheck AI", layout="wide")

# CSS Global
st.markdown("""
<style>
    [data-testid="stHeader"] {
        background-color: transparent;
        box-shadow: none;
    }
    .main-header {
        position: fixed;
        top: 0;
        left: 13rem;
        right: 0;
        background-color: #71b4f0;
        color: #000000;
        padding: 12px 20px;
        font-size: 24px;
        font-weight: 700;
        z-index: 999;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    [data-testid="stSidebar"][aria-expanded="false"] ~ div .main-header {
        left: 4rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #eee;
    }
    [data-testid="stSidebar"] .stButton>button {
        width: 100%;
        text-align: left;
        border: none;
        background: transparent;
        padding: 10px 15px;
        border-radius: 8px;
        color: #333;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #f0f2f6 !important;
    }
    [data-testid="stMain"] .stButton>button {
        background-color: #71b4f0 !important;
        color: white !important;
        border: none !important;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    [data-testid="stMain"] .stButton>button:hover {
        background-color: #5da3e0 !important;
    }
    [data-testid="stMain"] .stDownloadButton>button {
        background-color: #71b4f0 !important;
        color: white !important;
        border: none !important;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    [data-testid="stMain"] .stDownloadButton>button:hover {
        background-color: #5da3e0 !important;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .metric-label { font-size: 14px; color: #666; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1a1a1a; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource  
def load_essentials():
    MODEL_ID = "M9_801010_CLS_LR1e-4" 
    
    # Try-Except untuk mencegah NameError jika model gagal dimuat
    try:
        tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        bert_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
        bert_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model.to(device)
    except Exception:
        tokenizer, bert_model, device = None, None, None

    try:
        mlp_model = load_keras_model(f'models/{MODEL_ID}.h5', compile=False)
    except Exception:
        mlp_model = None

    try:
        threshold_data = np.load(f'models/best_threshold_{MODEL_ID}.npy')
        threshold = float(threshold_data)
    except:
        threshold = 0.5
        
    return tokenizer, bert_model, mlp_model, threshold, device

tokenizer, bert_model, mlp_model, THRESHOLD, device = load_essentials()

def get_gsheet_client():
    gc = gspread.service_account(filename='credentials.json')
    sh = gc.open("DatabaseJobCheck") 
    return sh.sheet1

def save_to_gsheet(data):
    try:
        ws = get_gsheet_client()
        # Menggunakan kunci sesuai permintaan Anda: nama_pekerjaan, tasks, prediction, date
        ws.append_row([data['nama_pekerjaan'], data['tasks'], data['prediction'], data['date']])
    except Exception as e:
        st.error(f"Gagal simpan ke Google Sheets: {e}")

def load_from_gsheet():
    try:
        ws = get_gsheet_client()
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

# Inisialisasi Session State
if "page" not in st.session_state:
    st.session_state.page = "Landing Page"  
if "test_result" not in st.session_state:
    st.session_state.test_result = None
if "y_pred" not in st.session_state:
    st.session_state.y_pred = None
if "text_check_result" not in st.session_state:
    st.session_state.text_check_result = None  
if "history_data" not in st.session_state:
    st.session_state.history_data = []  

def clean_text(text):
    if not text: return ""
    if "[SEP]" in str(text): return str(text)
    text = re.sub(r'^\s*\d+[a-zA-Z]?[\.\s)]+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\.\,]', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def get_bert_embedding_cls(text):
    if tokenizer is None or bert_model is None: return np.zeros((1, 768))
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def predict(title, tasks):
    if mlp_model is None: return "Error Load Model", 0.0, 0
    
    if "[SEP]" in str(tasks):
        text_input = f"{title} [SEP] {tasks}"
    else:
        text_input = f"{clean_text(title)} [SEP] {clean_text(tasks)}"
    
    embedding = get_bert_embedding_cls(text_input)
    prediction_prob = mlp_model.predict(embedding, verbose=0)
    prob_terotomatisasi = float(prediction_prob[0][1])
    label_idx = 1 if prob_terotomatisasi >= THRESHOLD else 0
    label_txt = "terotomatisasi" if label_idx == 1 else "tidak terotomatisasi"
    return label_txt, prob_terotomatisasi, label_idx

def paginate_dataframe(df, page_size, key):
    total_pages = max(len(df) // page_size + (1 if len(df) % page_size > 0 else 0), 1)
    col1, col2 = st.columns([1, 4])
    cp = col1.number_input(f"Halaman (Total {total_pages})", min_value=1, max_value=total_pages, step=1, key=key)
    return df.iloc[(cp - 1) * page_size:cp * page_size]

def highlight_ai_sentences(tasks):
    kw = ["data", "analisis", "sistem", "model",  "komputer", "perangkat", "mesin", "pengujian", "uji",  "rancang", "kembang", "laporan", "prediksi", "evaluasi", "monitoring", "otomatis", "digital", "pencatatan", "verifikasi", "efisiensi", "akurasi", "teknologi", "informasi", "prosedur", "metode", "spesifikasi"]   
    sentences = [s.strip() for s in tasks.replace("[SEP]", "\n").split("\n") if s.strip()]
    result = []
    for s in sentences:
        count = sum(1 for w in kw if re.search(r'\b' + w, s.lower()))
        result.append((s, count))
    return result

def skill_heatmap(tasks):
    kw = ["data", "analisis", "sistem", "model",  "komputer", "perangkat", "mesin", "pengujian", "uji",  "rancang", "kembang", "laporan", "prediksi", "evaluasi", "monitoring", "otomatis", "digital", "pencatatan", "verifikasi", "efisiensi", "akurasi", "teknologi", "informasi", "prosedur", "metode", "spesifikasi"]   
    tasks_lower = tasks.lower()
    scores = [len(re.findall(r'\b' + k, tasks_lower)) for k in kw]
    return pd.DataFrame({"skill": kw, "score": scores}).sort_values(by="score", ascending=False)

def normalize_text_for_comparison(text):
    """Normalize text: lowercase, hapus spasi, hapus punctuation"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Hapus punctuation
    text = re.sub(r'\s+', '', text)       # Hapus semua spasi
    return text

def check_duplicate_in_gsheet(nama_pekerjaan):
    """Cek apakah pekerjaan sudah ada di Google Sheets (dengan normalisasi)"""
    try:
        hist_df = load_from_gsheet()
        if hist_df.empty:
            return False
        
        normalized_new = normalize_text_for_comparison(nama_pekerjaan)
        
        for idx, row in hist_df.iterrows():
            normalized_existing = normalize_text_for_comparison(row['nama_pekerjaan'])
            if normalized_new == normalized_existing:
                return True
        
        return False
    except:
        return False

def are_jobs_similar(name1, name2, threshold=0.8):
    return SequenceMatcher(None, name1.lower().strip(), name2.lower().strip()).ratio() >= threshold

# --- LANDING PAGE ---
if st.session_state.page == "Landing Page":
    st.markdown("""
    <style>
        .stApp {
            background-color: #BAD1F5 !important;
        }
        .block-container {
            background-color: #ffffff !important;
            border-radius: 30px !important;
            padding: 60px 80px !important;
            margin-top: 10vh !important;
            max-width: 1050px !important;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
        }
        .landing-title {
            font-size: 24px;
            font-weight: 900;
            color: #000000;
            margin-bottom: 30px;
            line-height: 1.5;
        }
        .landing-text-normal {
            font-size: 22px;
            font-weight: normal;
            color: #000000;
            margin-bottom: 30px;
            line-height: 1.5;
        }
        .landing-text-bold {
            font-size: 18px;
            font-weight: bold;
            color: #000000;
            margin-bottom: 40px;
            line-height: 1.5;
        }
        div.stButton > button {
            background-color: #71b4f0 !important;
            color: #000000 !important;
            font-weight: 900 !important;
            font-size: 22px !important;
            border-radius: 8px !important;
            padding: 10px 40px !important;
            border: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    card_col1, card_col2 = st.columns([6, 4], gap="large")
    
    with card_col1:
        st.markdown("""
            <div class="landing-title">
                KLASIFIKASI PEKERJAAN DI INDONESIA YANG TERPENGARUH OLEH TEKNOLOGI<br>
                ARTIFICIAL INTELLIGENCE BERDASARKAN OCCUPATIONAL EXPOSURE SCORE<br>
                MENGGUNAKAN INDOBERT
            </div>
            <div class="landing-text-normal">
                KEZIA NATALIA<br>
                211402002
            </div>
            <div class="landing-text-bold">
                PROGRAM STUDI S1 TEKNOLOGI INFORMASI<br>
                FAKULTAS ILMU KOMPUTER DAN TEKNOLOGI INFORMASI<br>
                UNIVERSITAS SUMATERA UTARA<br>
                2026
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("START", key="start_btn"):
            st.session_state.page = "Jobs Testing File"
            st.rerun()
            
    with card_col2:
        try:
            st.image(r"D:\data kuliah\semester 8-9\skripsi\pro\logo.jpg", use_container_width=True)
        except:
            st.info("Logo tidak ditemukan")
            
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.markdown("""
<style>
    .block-container {
        padding-top: 6rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='color: #000080;'>JobCheck AI</h2>", unsafe_allow_html=True)
for p in ["Overview", "Cek Pekerjaan Teks", "Jobs Testing File", "Riwayat Deteksi"]:
    label = f"➤ {p}" if st.session_state.page == p else p
    if st.sidebar.button(label, key=f"sidebar_btn_{p}"):
        st.session_state.page = p
        st.rerun()

page = st.session_state.page
st.markdown(f"<div class='main-header'><div class='header-title'>{page}</div></div>", unsafe_allow_html=True)

def find_relevant_columns(df_columns):
    title_keywords = ['title', 'job', 'pekerjaan', 'posisi', 'occupation', 'name']
    task_keywords = ['tasks', 'task', 'tugas', 'deskripsi', 'description', 'jobdesc', 'kerja']
    found_title, found_tasks = None, None
    cols_lower = [c.lower() for c in df_columns]
    for kw in title_keywords:
        for i, col in enumerate(cols_lower):
            if kw in col: found_title = df_columns[i]; break
        if found_title: break
    for kw in task_keywords:
        for i, col in enumerate(cols_lower):
            if kw in col: found_tasks = df_columns[i]; break
        if found_tasks: break
    return found_title, found_tasks

# --- PAGE: OVERVIEW ---
if page == "Overview":
    try:
        # 1. Load data hasil prediksi model IndoBERT kamu
        df = pd.read_csv("data/hasil_klasifikasi_indobert.csv")
        
        # Bersihkan text separator agar rapi saat dibaca di tabel
        df['tasks_clean'] = df['tasks'].str.replace("[SEP]", ", ", regex=False)

        # 2. TAMPILKAN METRIC CARD (Berdasarkan kolom 'hasil_klasifikasi')
        m1, m2, m3 = st.columns(3)
        with m1: 
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Pekerjaan</div><div class='metric-value'>{len(df)}</div></div>", unsafe_allow_html=True)
        with m2: 
            total_terotomasi = len(df[df['hasil_klasifikasi'] == 'Terotomatisasi'])
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Terotomasi AI (Model)</div><div class='metric-value'>{total_terotomasi}</div></div>", unsafe_allow_html=True)
        with m3: 
            total_aman = len(df[df['hasil_klasifikasi'] == 'Tidak Terotomatisasi'])
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Tidak Terotomatisasi (Model)</div><div class='metric-value'>{total_aman}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 3. CONTAINER DAFTAR PEKERJAAN & FILTER
        with st.container(border=True):
            st.subheader("Daftar Hasil Klasifikasi Pekerjaan")
            
            f1, f2 = st.columns([2, 1])
            search_query = f1.text_input("🔍 Cari Nama Pekerjaan", placeholder="Ketik nama pekerjaan...")
            # Sesuaikan opsi selectbox dengan nilai teks di kolom hasil_klasifikasi
            filter_label = f2.selectbox("Filter Status Otomasi", ["Semua", "Terotomatisasi", "Tidak Terotomatisasi"])

            df_display = df.copy()
            
            # Filter berdasarkan Pencarian Nama Pekerjaan
            if search_query: 
                df_display = df_display[df_display['title'].str.contains(search_query, case=False, na=False)]
            
            # Filter berdasarkan Dropdown Status Klasifikasi Model
            if filter_label != "Semua": 
                df_display = df_display[df_display['hasil_klasifikasi'] == filter_label]

            # Format kolom score_prediksi menjadi persentase agar siap tampil
            df_display['score_prediksi_fmt'] = df_display['score_prediksi'].map(lambda x: f"{x:.2%}")

            # Menyusun kolom yang dipilih (memasukkan oidn_code jika ada)
            cols_to_select = ['title', 'tasks_clean', 'exposure_score', 'score_prediksi_fmt', 'hasil_klasifikasi']
            if 'oidn_code' in df_display.columns:
                cols_to_select = ['oidn_code'] + cols_to_select
            
            # Rename nama kolom agar tampak profesional di interface website
            rename_dict = {
                'oidn_code': 'OIDN_CODE',
                'title': 'Nama Pekerjaan', 
                'tasks_clean': 'Deskripsi Tugas', 
                'exposure_score': 'Skor OES',
                'score_prediksi_fmt': 'Confidence AI',
                'hasil_klasifikasi': 'Status Otomasi'
            }
            
            df_final = df_display[cols_to_select].rename(columns=rename_dict)
            
            # Tampilkan tabel data dengan pagination milikmu
            st.dataframe(paginate_dataframe(df_final, 10, "ov_page"), use_container_width=True, hide_index=True)

        # 4. VISUALISASI GRAFIK (CHART)
        ch1, ch2 = st.columns(2)
        with ch1:
            with st.container(border=True):
                st.subheader("Kategori Dampak AI (Prediksi Model)")
                # Hitung value counts langsung dari kolom hasil_klasifikasi
                lc = df['hasil_klasifikasi'].value_counts().reset_index()
                lc.columns = ['Status Otomasi', 'Jumlah']
                
                # Gambar Donut Chart hasil model IndoBERT
                fig_pie = px.pie(lc, names='Status Otomasi', values='Jumlah', hole=0.7, 
                                 color_discrete_sequence=["#4F6DFF", "#E0E7FF"])
                st.plotly_chart(fig_pie, use_container_width=True)
                
        with ch2:
            with st.container(border=True):
                st.subheader("Top 10 Pekerjaan Rentan AI")
                # Mengurutkan berdasarkan skor exposure tertinggi untuk melihat pekerjaan ter-expose
                top10 = df.sort_values(by='exposure_score', ascending=False).head(10)
                
                # Gambar Bar Chart horizontal
                fig_bar = px.bar(top10, x='exposure_score', y='title', orientation='h', 
                                 labels={'exposure_score': 'Skor OES', 'title': 'Pekerjaan'},
                                 color_discrete_sequence=["#4F6DFF"])
                # Membalik urutan y-axis agar peringkat #1 berada di paling atas chart
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
                
    except FileNotFoundError:
        st.error("File 'data/hasil_klasifikasi_indobert.csv' tidak ditemukan. Pastikan file hasil prediksi model Anda sudah diexport ke folder data.")

# --- PAGE: CEK TEKS ---
elif page == "Cek Pekerjaan Teks":
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<h4 style='text-align:center'>Selamat Datang di JobCheck</h4>",unsafe_allow_html=True)
        st.markdown("<p style='text-align:center'>Klasifikasi pekerjaan yang mungkin terpengaruh oleh teknologi AI</p>",unsafe_allow_html=True)
        st.markdown("<p style='text-align:center'>Masukkan 1 Pekerjaan dan Deskripsi Tugasnya</p>",unsafe_allow_html=True)
        
        job_name = st.text_input("Nama Pekerjaan", placeholder="Contoh: Akuntan", key="input_job_name", on_change=lambda: st.session_state.update({"text_check_result": None}))
        job_desc = st.text_area("Deskripsi Task", placeholder="Contoh: Menginput data transaksi harian...", height=150, key="input_job_desc", on_change=lambda: st.session_state.update({"text_check_result": None}))
        
        # Tampilkan counter karakter
        char_count = len(job_desc)
        if char_count > 700:
            st.error(f"⚠️ Deskripsi melebihi 700 karakter ({char_count}/700). Hanya 700 karakter pertama yang akan diproses.")
            job_desc_processed = job_desc[:700]
        else:
            st.warning(f"Jumlah Karakter: {char_count}/700")
            job_desc_processed = job_desc
        
        if st.button("Mulai Analisis"):
            if not job_name.strip() or not job_desc_processed.strip():
                st.error("⚠️ Silahkan masukkan nama pekerjaan dan tugasnya")
            else:
                label_txt, conf, lbl_idx = predict(job_name, job_desc_processed)
                st.session_state.text_check_result = {"label": label_txt, "conf": conf, "desc": job_desc_processed}
                
                new_entry = {
                    "nama_pekerjaan": job_name, 
                    "tasks": job_desc_processed, 
                    "prediction": label_txt.title(), 
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # CEK DUPLIKAT SEBELUM SAVE KE GSHEET
                is_duplicate = check_duplicate_in_gsheet(job_name)
                
                if not is_duplicate:
                    # Jika tidak ada duplikat, simpan ke Google Sheets
                    save_to_gsheet(new_entry)
                    st.success("✅ Data berhasil disimpan ke riwayat")
                else:
                    st.warning("Pekerjaan ini sudah pernah dicek sebelumnya. Data tidak disimpan ulang.", icon="🚨")
                
                # Update history session state
                history = st.session_state.history_data
                match_index = -1
                
                for i, h in enumerate(history):
                    if are_jobs_similar(h['nama_pekerjaan'], job_name, threshold=0.8):
                        match_index = i
                        break
                
                if match_index != -1:
                    st.session_state.history_data[match_index] = new_entry 
                else:
                    st.session_state.history_data.insert(0, new_entry)
        
        if st.session_state.text_check_result:
            res = st.session_state.text_check_result
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Hasil", res["label"].title())
            col_res2.metric("Confidence Score", f"{res['conf']:.2%}")
            
            if res["conf"] >= THRESHOLD: 
                st.success("Pekerjaan ini memiliki kemungkinan Terotomatisasi AI")
            else: 
                st.info("Pekerjaan ini relatif aman dari otomatisasi AI")
            
            st.subheader("Analisis Task")
            for sent, score in highlight_ai_sentences(res["desc"]):
                if score > 0:
                    highlighted_sent = sent
                    kw = ["data", "analisis", "sistem", "model",  "komputer", "perangkat", "mesin", "pengujian", "uji",  "rancang", "kembang", "laporan", "prediksi", "evaluasi", "monitoring", "otomatis", "digital", "pencatatan", "verifikasi", "efisiensi", "akurasi", "teknologi", "informasi", "prosedur", "metode", "spesifikasi"]   
                    for keyword in kw:
                        if keyword in sent.lower():
                            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                            highlighted_sent = pattern.sub(f"**{keyword}**", highlighted_sent)
                    
                    st.markdown(f"- 🔴 {highlighted_sent}")
                else:
                    st.markdown(f"- ⚪ {sent}")
            
            fig, ax = plt.subplots(figsize=(8, 2))
            sns.heatmap(skill_heatmap(res["desc"]).set_index("skill").T, cmap="Blues", annot=True, cbar=False, ax=ax)
            st.pyplot(fig)

# --- PAGE: TESTING FILE ---
elif page == "Jobs Testing File":
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<h4 style='text-align:center'>Selamat Datang di JobCheck</h4>",unsafe_allow_html=True)
        st.markdown("<p style='text-align:center'>Upload File untuk Testing Model</p>",unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV Data Testing", type=["csv"], key="file_checker", 
                                     on_change=lambda: st.session_state.update({"test_result": None, "y_pred": None}))
    
    if uploaded_file:
        data_test = pd.read_csv(uploaded_file)
        col_title, col_tasks = find_relevant_columns(data_test.columns)
        st.session_state.col_title = col_title
        st.session_state.col_tasks = col_tasks
        
        if not col_title or not col_tasks:
            st.error(f"❌ Gagal mendeteksi kolom. Pastikan CSV memiliki kolom Nama Pekerjaan dan Tasks. Kolom yang ditemukan: {list(data_test.columns)}")
        else:
            st.success(f"✅ Kolom terdeteksi: '{col_title}' sebagai Pekerjaan & '{col_tasks}' sebagai Deskripsi Tugas Pekerjaan")
            
            if st.button("Upload dan Mulai Testing"):
                progress = st.progress(0)
                y_pred, y_p_text, conf_scores = [], [], []
                total = len(data_test)

                for i, row in data_test.iterrows():
                    txt, conf, lbl = predict(row[col_title], row[col_tasks])
                    y_p_text.append(txt)
                    y_pred.append(lbl)
                    conf_scores.append(conf)
                    progress.progress((i + 1) / total)

                data_test['Nama Pekerjaan'] = data_test[col_title].astype(str).str.lower()
                
                def format_display_tasks(text):
                    t = str(text)
                    return t.replace("[SEP]", ", ").lower() if "[SEP]" in t else clean_text(t)

                data_test['Tasks_Display'] = data_test[col_tasks].apply(format_display_tasks)
                data_test['Label Predict'] = y_p_text
                data_test['Confidence Score'] = [f"{c:.2%}" for c in conf_scores]
                data_test['confidence_numeric'] = conf_scores

                st.session_state.test_result = data_test
                st.session_state.y_pred = y_pred

    if st.session_state.test_result is not None and st.session_state.y_pred is not None:
        dt = st.session_state.test_result
        has_ground_truth = 'label' in dt.columns
        cols_to_show = ['Nama Pekerjaan', 'Tasks_Display']
        if has_ground_truth: cols_to_show.append('label')
        cols_to_show.extend(['Label Predict', 'Confidence Score'])

        if has_ground_truth:
            st.subheader("📊 Evaluasi Model")
            yt = dt['label'].values
            st.metric("Akurasi Model", f"{accuracy_score(yt, st.session_state.y_pred) * 100:.2f}%")
            
            col_left, col_mid, col_right = st.columns([1, 1, 1]) 
            with col_mid:
                fig, ax = plt.subplots(figsize=(3, 2))
                sns.heatmap(confusion_matrix(yt, st.session_state.y_pred), annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 8})
                ax.tick_params(labelsize=7)
                st.pyplot(fig)
        else:
            st.info("Data tidak memiliki ground truth. Sistem hanya menampilkan hasil prediksi.")

        # Penamaan kolom 
        cols_rename_map = {'Nama Pekerjaan': 'nama pekerjaan', 'Tasks_Display': 'Taks'}
        if has_ground_truth: cols_rename_map['label'] = 'Ground Truth'
        
        st.dataframe(dt[cols_to_show].rename(columns=cols_rename_map), use_container_width=True, hide_index=True)

        st.markdown("### 📊 Visualisasi Hasil Prediksi")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Kategori Dampak AI")
            pie_df = dt['Label Predict'].value_counts().reset_index()
            pie_df.columns = ['Kategori', 'Jumlah']
            st.plotly_chart(px.pie(pie_df, names='Kategori', values='Jumlah', hole=0.6), use_container_width=True)
        with col2:
            st.subheader("Top 10 Pekerjaan Rentan AI")
            top_df = dt.copy()
            top_df['confidence_numeric'] = top_df['confidence_numeric'].astype(float)
            top_df = top_df.sort_values(by='confidence_numeric', ascending=False).head(10)
            fig_bar = px.bar(top_df, x='confidence_numeric', y='Nama Pekerjaan', orientation='h',
                            labels={'confidence_numeric': 'Confidence Score', 'Nama Pekerjaan': 'Pekerjaan'},
                            color_discrete_sequence=["#4F6DFF"])
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.markdown("### 💾 Simpan Hasil Prediksi")
        col_title, col_tasks = st.session_state.col_title, st.session_state.col_tasks
        export_df = dt[[col_title, col_tasks]].copy()
        if 'label' in dt.columns: export_df['Ground Truth'] = dt['label']
        export_df['Label Predict'] = dt['Label Predict']
        export_df['Confidence Score'] = dt['Confidence Score']

        file_format = st.selectbox("Pilih format file", ["CSV", "Excel"])
        if file_format == "CSV":
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="⬇️ Download CSV", data=csv, file_name="hasil_prediksi_jobcheck.csv", mime='text/csv')
        else:
            output = BytesIO()
            try:
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer: export_df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
            except ModuleNotFoundError:
                with pd.ExcelWriter(output, engine='openpyxl') as writer: export_df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
            st.download_button(label="⬇️ Download Excel", data=output.getvalue(), file_name="hasil_prediksi_jobcheck.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# --- PAGE: RIWAYAT DETEKSI ---
elif page == "Riwayat Deteksi":
    st.markdown("<h3 style='text-align:center;'>Riwayat Deteksi</h3>", unsafe_allow_html=True)
    
    # Inisialisasi session state untuk filter
    if "riwayat_start_date" not in st.session_state:
        st.session_state.riwayat_start_date = None
    if "riwayat_end_date" not in st.session_state:
        st.session_state.riwayat_end_date = None
    if "riwayat_search" not in st.session_state:
        st.session_state.riwayat_search = ""
    
    # Ambil data
    hist_df = load_from_gsheet()
    
    if hist_df.empty:
        st.info("Belum ada riwayat pengecekan di Sistem JobCheck AI.")
    else:
        # Konversi kolom 'date' ke format datetime agar bisa difilter
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        
        # DEDUPLIKASI: Hapus duplikat berdasarkan nama pekerjaan dan tasks (tanpa spasi)
        hist_df['_nama_clean'] = hist_df['nama_pekerjaan'].str.lower().str.strip()
        hist_df['_tasks_clean'] = hist_df['tasks'].str.lower().str.replace(' ', '', regex=False)
        hist_df = hist_df.drop_duplicates(subset=['_nama_clean', '_tasks_clean'], keep='last')
        hist_df = hist_df.drop(columns=['_nama_clean', '_tasks_clean'])
        
        # Set default date jika belum ada
        if st.session_state.riwayat_start_date is None:
            st.session_state.riwayat_start_date = hist_df['date'].min().date()
        if st.session_state.riwayat_end_date is None:
            st.session_state.riwayat_end_date = hist_df['date'].max().date()
        
        # UI Filter Tanggal
        f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
        with f_col1: 
            start_date = st.date_input("Dari Tanggal", value=st.session_state.riwayat_start_date, key="start_date_input")
            st.session_state.riwayat_start_date = start_date
        with f_col2: 
            end_date = st.date_input("Ke Tanggal", value=st.session_state.riwayat_end_date, key="end_date_input")
            st.session_state.riwayat_end_date = end_date
        with f_col3: 
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Reset Filter", key="reset_filter_btn"):
                # Reset semua filter
                st.session_state.riwayat_start_date = hist_df['date'].min().date()
                st.session_state.riwayat_end_date = hist_df['date'].max().date()
                st.session_state.riwayat_search = ""
                st.rerun()
        
        # Logika Filter Data berdasarkan tanggal
        mask = (hist_df['date'].dt.date >= start_date) & (hist_df['date'].dt.date <= end_date)
        filtered_df = hist_df.loc[mask]
        
        # Tampilkan Tabel
        display_hist = filtered_df.rename(columns={
            "nama_pekerjaan": "Nama Pekerjaan",
            "tasks": "Tasks",
            "prediction": "Prediction",
            "date": "Date"
        })
        
        # SEARCH INPUT DAN BUTTON SEARCH
        s_col1, s_col2 = st.columns([4, 1])
        with s_col1:
            search = st.text_input("Search:", placeholder="Cari nama pekerjaan di riwayat...", value=st.session_state.riwayat_search, key="search_input")
            st.session_state.riwayat_search = search
        
        with s_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.button("🔍 Cari", key="search_btn")
        
        # Filter berdasarkan search
        if search:
            display_hist = display_hist[display_hist['Nama Pekerjaan'].str.contains(search, case=False)]
        
        # CEK HASIL SETELAH SEARCH DAN TAMPILKAN ALERT
        if display_hist.empty:
            st.warning("⚠️ Pekerjaan tersebut belum pernah di deteksi, silahkan melakukan cek pekerjaan terlebih dahulu")
        else:
            # CSS dan HTML Table
            st.markdown("""
            <style>
                .riwayat-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }
                .riwayat-table thead th {
                    background-color: #f0f2f6;
                    padding: 12px;
                    text-align: left;
                    border: 1px solid #e0e0e0;
                    font-weight: 600;
                }
                .riwayat-table tbody td {
                    padding: 12px;
                    border: 1px solid #e0e0e0;
                    word-wrap: break-word;
                    white-space: normal;
                    overflow-wrap: break-word;
                }
                .riwayat-table tbody tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .riwayat-table tbody tr:hover {
                    background-color: #f0f0f0;
                }
                .col-tasks {
                    max-width: 500px;
                    word-break: break-word;
                    white-space: normal;
                }
                .col-nama {
                    max-width: 150px;
                }
                .col-prediction {
                    max-width: 150px;
                }
                .col-date {
                    max-width: 150px;
                    white-space: nowrap;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Generate HTML table
            html_table = "<table class='riwayat-table'><thead><tr>"
            html_table += "<th>Nama Pekerjaan</th><th>Tasks</th><th>Prediction</th><th>Date</th></tr></thead><tbody>"
            
            for idx, row in display_hist[["Nama Pekerjaan", "Tasks", "Prediction", "Date"]].iterrows():
                html_table += f"<tr>"
                html_table += f"<td class='col-nama'>{row['Nama Pekerjaan']}</td>"
                html_table += f"<td class='col-tasks'>{row['Tasks']}</td>"
                html_table += f"<td class='col-prediction'>{row['Prediction']}</td>"
                html_table += f"<td class='col-date'>{row['Date']}</td>"
                html_table += f"</tr>"
            
            html_table += "</tbody></table>"
            st.markdown(html_table, unsafe_allow_html=True)