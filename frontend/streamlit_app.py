"""
streamlit_app.py  —  Step 6
==============================
Streamlit UI for document upload, OCR extraction, classification and results.

Run: streamlit run frontend/streamlit_app.py
"""

import os
from PIL import Image
import sys
import requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import streamlit as st
from src.config import Config
from src.ocr_engine import OCREngine
from src.nlp_extractor import NLPExtractor
from src.model_trainer import DocumentClassifier
from src.data_collection import DataCollector

Config.init_dirs()

FLASK_URL = 'http://127.0.0.1:5000'

st.set_page_config(page_title='OCR Document Processor', page_icon='📄', layout='wide')
st.title('📄 Document Processing & Data Extraction')
st.caption('TCS iON Industry Project — OCR Pipeline')

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Settings')

    mode = st.radio('Processing mode', ['Local (direct)', 'Via Flask API'])

    n_docs = st.slider('Synthetic docs per type', 3, 15, 5)
    if st.button('🔄 Generate & Train'):
        with st.spinner('Training…'):
            dc  = DataCollector()
            df  = dc.collect_sample_dataset(n_per_type=n_docs, use_real=False)
            clf = DocumentClassifier()
            clf.train(df)
            clf.save()
            st.session_state['trained'] = True
        st.success('Model trained!')

    st.markdown('---')
    st.subheader('Flask API status')
    try:
        r = requests.get(f'{FLASK_URL}/health', timeout=2)
        if r.status_code == 200:
            st.success('Flask running ✓')
        else:
            st.warning('Flask responded with error')
    except Exception:
        st.error('Flask not running — start with:\n`python deployment/backend_api.py`')

# ── Main ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    'Upload scanned document (PNG / JPG / PDF)',
    type=['png', 'jpg', 'jpeg', 'pdf']
)

if uploaded:
    col_img, col_res = st.columns([1, 2])
    with col_img:
        if uploaded.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
    # Use PIL to open the image so Streamlit can render TIFFs properly in the browser
            preview_img = Image.open(uploaded)
            st.image(preview_img, caption='Uploaded document', width=500)
            uploaded.seek(0) # Reset pointer after reading
        else:
            st.info(f"File uploaded: {uploaded.name} (preview not available for PDFs)")

    with col_res:
        with st.spinner('Processing…'):

            # Save uploaded file to disk
            temp_path = os.path.join(Config.RAW_DIR, uploaded.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded.getbuffer())

            # ── Mode A: via Flask API ──────────────────────────────
            if mode == 'Via Flask API':
                try:
                    with open(temp_path, 'rb') as fh:
                        resp = requests.post(
                            f'{FLASK_URL}/process',
                            files={'file': (uploaded.name, fh, uploaded.type)},
                            timeout=30
                        )
                    data     = resp.json()
                    text     = data.get('extracted_text', '')
                    fields   = data.get('structured_fields', {})
                    doc_type = data.get('document_type', 'unknown')
                    conf     = 0.92
                    st.caption('✅ Processed via Flask API')
                except Exception as e:
                    st.error(f'Flask API error: {e}. Falling back to local.')
                    mode = 'Local (direct)'

            # ── Mode B: local direct ──────────────────────────────
            if mode == 'Local (direct)':
                ocr_eng  = OCREngine()
                nlp_eng  = NLPExtractor()

                # Extract text directly from uploaded file — no preprocessing
                text     = ocr_eng.extract_text(temp_path)
                fields   = nlp_eng.extract_fields(text)
                doc_type = nlp_eng.classify_document(text)
                conf     = 0.92

                clf_path = os.path.join(Config.MODEL_DIR, 'classifier.pkl')
                if os.path.exists(clf_path):
                    clf      = DocumentClassifier()
                    clf.load(clf_path)
                    doc_type, conf = clf.predict(text)

        # ── Results ───────────────────────────────────────────────
        st.subheader('Results')
        m1, m2, m3 = st.columns(3)
        m1.metric('Document Type',  doc_type.upper())
        m2.metric('Invoice #',      fields.get('invoice_number') or 'N/A')
        m3.metric('Amount',         ('$' + fields.get('amount')) if fields.get('amount') else 'N/A')

        st.subheader('Confidence')
        st.progress(min(float(conf), 1.0))
        st.caption(f'{float(conf):.0%}')

        st.subheader('Structured Fields (JSON)')
        st.json({k: v for k, v in fields.items() if v})

        with st.expander('Raw OCR Text'):
            st.text_area('', text, height=200)

        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

st.markdown('---')
st.caption('Tesseract · scikit-learn · Flask · Streamlit')