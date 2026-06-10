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
    kw = ["data", "analisis", "laporan", "otomatis", "mengolah", "memproses", "administrasi", "entry", "menghitung", "prediksi", "rutin", "arsip", "verifikasi", "digital", "perangkat lunak", "evaluasi", "monitoring", "dokumen", "pencatatan", "klasifikasi"]
    sentences = [s.strip() for s in tasks.replace("[SEP]", "\n").split("\n") if s.strip()]
    result = []
    for s in sentences:
        count = sum(1 for w in kw if re.search(r'\b' + w, s.lower()))
        result.append((s, count))
    return result

def skill_heatmap(tasks):
    kw = ["data", "analisis", "laporan", "otomatis", "mengolah", "memproses", "administrasi", "entry", "menghitung", "prediksi", "rutin", "arsip", "verifikasi", "digital", "perangkat lunak", "evaluasi", "monitoring", "dokumen", "pencatatan", "klasifikasi"]
    tasks_lower = tasks.lower()
    scores = [len(re.findall(r'\b' + k, tasks_lower)) for k in kw]
    return pd.DataFrame({"skill": kw, "score": scores}).sort_values(by="score", ascending=False)

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
        df = pd.read_csv("data/exposure_job.csv")
        df['tasks_clean'] = df['tasks'].str.replace("[SEP]", ", ", regex=False)

        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Pekerjaan</div><div class='metric-value'>{len(df)}</div></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Terotomasi AI</div><div class='metric-value'>{len(df[df['label']==1])}</div></div>", unsafe_allow_html=True)
        with m3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Tidak Teromatisasi AI</div><div class='metric-value'>{len(df[df['label']==0])}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("Daftar Pekerjaan")
            f1, f2 = st.columns([2, 1])
            search_query = f1.text_input("🔍 Cari Nama Pekerjaan", placeholder="Ketik nama pekerjaan...")
            filter_label = f2.selectbox("Filter Klasifikasi", ["Semua", "terotomatisasi", "tidak terotomatisasi"])

            df_display = df.copy()
            df_display['klasifikasi'] = df_display['label'].map({1: "terotomatisasi", 0: "tidak terotomatisasi"})
            
            if search_query: 
                df_display = df_display[df_display['title'].str.contains(search_query, case=False, na=False)]
            if filter_label != "Semua": 
                df_display = df_display[df_display['klasifikasi'] == filter_label]

            # Pilih kolom yang dibutuhkan saja SEBELUM direname untuk menghindari duplikasi
            cols_to_select = ['title', 'tasks_clean', 'klasifikasi', 'exposure_score']
            if 'oidn_code' in df_display.columns:
                cols_to_select = ['oidn_code'] + cols_to_select
            
            df_final = df_display[cols_to_select].rename(columns={
                'title': 'Nama Pekerjaan', 
                'tasks_clean': 'Tasks', 
                'klasifikasi': 'Klasifikasi',
                'exposure_score': 'Skor'
            })
            
            st.dataframe(paginate_dataframe(df_final, 10, "ov_page"), use_container_width=True, hide_index=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            with st.container(border=True):
                st.subheader("Kategori Dampak AI pada Pekerjaan")
                lc = df['label'].value_counts().reset_index()
                lc.columns = ['label', 'jumlah']
                lc['label'] = lc['label'].map({1: "Terotomasi AI", 0: "Tidak Terotomatisasi AI"})
                st.plotly_chart(px.pie(lc, names='label', values='jumlah', hole=0.7, color_discrete_sequence=["#4F6DFF", "#E0E7FF"]), use_container_width=True)
        with ch2:
            with st.container(border=True):
                st.subheader("Top Pekerjaan Rentan AI")
                top10 = df.sort_values(by='exposure_score', ascending=False).head(10)
                st.plotly_chart(px.bar(top10, x='exposure_score', y='title', orientation='h', color_discrete_sequence=["#4F6DFF"]), use_container_width=True)
    except FileNotFoundError:
        st.error("File 'data/exposure_job.csv' tidak ditemukan.")

# --- PAGE: CEK TEKS ---
elif page == "Cek Pekerjaan Teks":
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<h4 style='text-align:center'>Selamat Datang di JobCheck</h4>",unsafe_allow_html=True)
        st.markdown("<p style='text-align:center'>Klasifikasi pekerjaan yang mungkin terpengaruh oleh teknologi AI</p>",unsafe_allow_html=True)
        st.markdown("<p style='text-align:center'>Masukkan 1 Pekerjaan dan Deskripsi Tugasnya</p>",unsafe_allow_html=True)
        
        job_name = st.text_input("Nama Pekerjaan", placeholder="Contoh: Akuntan", key="input_job_name", on_change=lambda: st.session_state.update({"text_check_result": None}))
        job_desc = st.text_area("Deskripsi Task", placeholder="Contoh: Menginput data transaksi harian...", height=150, key="input_job_desc", on_change=lambda: st.session_state.update({"text_check_result": None}))

        if st.button("Mulai Analisis"):
            if not job_name.strip() or not job_desc.strip():
                st.warning("⚠️ Silahkan masukkan nama pekerjaan dan tugasnya")
            else:
                label_txt, conf, lbl_idx = predict(job_name, job_desc)
                st.session_state.text_check_result = {"label": label_txt, "conf": conf, "desc": job_desc}

                history = st.session_state.history_data
                match_index = -1
                
                for i, h in enumerate(history):
                    if are_jobs_similar(h['nama'], job_name, threshold=0.8):
                        match_index = i
                        break

                new_entry = {
                    "nama_pekerjaan": job_name, 
                    "tasks": job_desc, 
                    "prediction": label_txt.title(), 
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                save_to_gsheet(new_entry)
                st.session_state.history_data.insert(0, new_entry)

                if match_index != -1:
                    st.session_state.history_data[match_index] = new_entry 
                else:
                    st.session_state.history_data.insert(0, new_entry)

        if st.session_state.text_check_result:
            res = st.session_state.text_check_result
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Hasil", res["label"].title())
            col_res2.metric("Confidence Score", f"{res['conf']:.2%}")
            if res["conf"] >= THRESHOLD: st.success("Pekerjaan ini memiliki kemungkinan Terotomatisasi AI")
            else: st.info("Pekerjaan ini relatif aman dari otomatisasi AI")

            st.subheader("Analisis Task")
            for sent, score in highlight_ai_sentences(res["desc"]):
                st.markdown(f"- {'🔴 **' + sent + '**' if score > 0 else '⚪ ' + sent}")

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
            st.subheader("Top Pekerjaan Rentan AI")
            top_df = dt.copy()
            top_df['confidence_numeric'] = top_df['confidence_numeric'].astype(float)
            top_df = top_df.sort_values(by='confidence_numeric', ascending=False).head(10)
            st.plotly_chart(px.bar(top_df, x='confidence_numeric', y='Nama Pekerjaan', orientation='h'), use_container_width=True)

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
        
        # UI Filter Tanggal
        f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
        with f_col1: 
            start_date = st.date_input("Dari Tanggal", value=hist_df['date'].min())
        with f_col2: 
            end_date = st.date_input("Ke Tanggal", value=hist_df['date'].max())
        with f_col3: 
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Reset Filter"):
                st.rerun()
        
        # Logika Filter Data
        mask = (hist_df['date'].dt.date >= start_date) & (hist_df['date'].dt.date <= end_date)
        filtered_df = hist_df.loc[mask]
        
        # Tampilkan Tabel
        display_hist = filtered_df.rename(columns={
            "nama_pekerjaan": "Nama Pekerjaan",
            "tasks": "Tasks",
            "prediction": "Prediction",
            "date": "Date"
        })
        
        # SEARCH INPUT (SEBELUM CEK KOSONG)
        search = st.text_input("Search:", placeholder="Cari nama pekerjaan di riwayat...")
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