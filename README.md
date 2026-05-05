# Document Processing & Data Extraction — OCR Computer Vision
**TCS iON Industry Project** | All issues from fix map resolved

---

## What's Fixed vs Previous Version
| Step | Fix Applied |
|------|------------|
| Step 1 | 7 doc types (invoice, contract, report, email, memo, receipt, PO) |
| Step 1 | RVL-CDIP real dataset via HuggingFace (`use_real=True`) |
| Step 2 | Correct pipeline order: resize→denoise→brightness→CLAHE→deskew→binarize→artifacts |
| Step 2 | `cv2.resize()` added at pipeline start |
| Step 2 | CLAHE contrast enhancement (`cv2.createCLAHE`) |
| Step 2 | Both OTSU + Adaptive thresholding (combined via bitwise AND) |
| Step 2 | Artifact removal upgraded to 3×3 kernel + MORPH_CLOSE |
| Step 2 | Before/after visual proof saved to `reports/before_after_preprocessing.png` |
| Step 3 | `NLPExtractor` fully defined — no more crash |
| Step 3 | OCR error correction dict + regex post-fixes (`correct_ocr_errors()`) |
| Step 3 | LayoutLM fine-tuning class (`LayoutLMClassifier`) |
| Step 4 | A/B test: Tesseract PSM 6 vs PSM 3 (`ocr.ab_test_psm()`) |
| Step 4 | A/B test: RandomForest vs NaiveBayes (`clf.ab_test_classifiers()`) |
| Step 4 | Flask runs via `threading` — proven alive with `/health` check |
| Step 4 | Benchmark on real vs synthetic data separately |
| Step 5 | Streamlit launched via `subprocess` |
| Step 5 | Streamlit calls Flask `/process` endpoint (integration complete) |
| Step 5 | PDF upload no longer crashes (`type` check + proc_path always `.png`) |

---

## Folder Structure
```
OCR_Document_Processing/
├── main_pipeline.py          ← run this — executes everything
├── requirements.txt
├── README.md
├── src/
│   ├── config.py             ← paths, 7 doc types
│   ├── data_collection.py    ← synthetic (7 types) + RVL-CDIP loader
│   ├── preprocessing.py      ← fixed order, CLAHE, adaptive, before/after
│   ├── ocr_engine.py         ← Tesseract + PSM A/B test
│   ├── nlp_extractor.py      ← NLPExtractor + OCR correction dict
│   ├── model_trainer.py      ← RF + NaiveBayes A/B + LayoutLM fine-tune
│   ├── evaluator.py          ← all metrics + source benchmarking
│   └── report_generator.py  ← PROJECT_REPORT.md with A/B results
├── tests/
│   ├── test_scenarios.py     ← 8 scenarios + pytest suite
│   └── test_scenarios.csv    ← auto-generated
├── deployment/
│   └── backend_api.py        ← Flask API + start_in_thread()
├── frontend/
│   └── streamlit_app.py      ← Streamlit UI, calls Flask, fixed PDF crash
├── notebooks/
│   └── OCR_Document_Processing_Complete.ipynb
├── data/raw/                 ← generated at runtime
├── data/processed/           ← preprocessed images
├── models/classifier.pkl     ← saved after training
└── reports/
    ├── before_after_preprocessing.png
    ├── confusion_matrix.png
    └── PROJECT_REPORT.md
```

---

## Setup

### 1 — Install Tesseract (system)
```bash
# Ubuntu
sudo apt-get install -y tesseract-ocr

# macOS
brew install tesseract

# Windows — download installer:
# https://github.com/UB-Mannheim/tesseract/wiki
# Then set in src/config.py:
# TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### 2 — Install Poppler (for PDF support)
```bash
# Ubuntu
sudo apt-get install -y poppler-utils

# macOS
brew install poppler

# Windows — download from:
# https://github.com/oschwartz10612/poppler-windows/releases
# Add bin/ folder to PATH or pass poppler_path= in preprocessing.py
```

### 3 — Python environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
```

---

## Running

### Full pipeline (all steps)
```bash
python main_pipeline.py
```
This runs: data collection → preprocessing → OCR A/B → NLP → training →
classifier A/B → evaluation → test scenarios → Flask (threaded) → report → zip.

### Individual steps
```bash
# Step 5 — Streamlit UI
streamlit run frontend/streamlit_app.py

# Step 4 — Flask API (standalone)
python deployment/backend_api.py

# Test Flask
curl http://localhost:5000/health
curl -X POST http://localhost:5000/process -F "file=@data/raw/invoice_1.png"

# Tests
pytest tests/test_scenarios.py -v

# Step 3 — LayoutLM fine-tune (requires torch + transformers)
python -c "
import pandas as pd
from src.model_trainer import LayoutLMClassifier
df  = pd.read_csv('data/metadata.csv')
clf = LayoutLMClassifier(num_labels=7)
clf.train(df, epochs=2)
"
```

### Use real RVL-CDIP dataset
```bash
python -c "
from main_pipeline import run_pipeline
run_pipeline(use_real_data=True)
"
```

### Flask via threading (from notebook)
```python
from deployment.backend_api import start_in_thread
start_in_thread(port=5000)
# Flask now running in background — call /process from Streamlit or curl
```

---

## A/B Tests Explained
| Test | Method A | Method B | How to run |
|------|----------|----------|------------|
| OCR quality | PSM 6 (uniform text) | PSM 3 (auto segment) | `ocr.ab_test_psm(image_path)` |
| Classifier | RandomForest | NaiveBayes | `clf.ab_test_classifiers(meta_df)` |

Results printed to console + saved in `reports/PROJECT_REPORT.md`.

---

## LayoutLM Fine-Tuning
Requires `pip install torch transformers`.  
Uses `microsoft/layoutlm-base-uncased` in text-only mode (no bounding boxes needed).  
Falls back to `DocumentClassifier` (RF) automatically if torch not installed.
