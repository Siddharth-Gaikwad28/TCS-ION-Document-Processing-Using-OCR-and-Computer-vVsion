# 📄 AI-Powered Document Processing System
### OCR · NLP · Machine Learning · Flask · Streamlit

> **TCS iON AIP 135 Curriculum Project**  
> F.Y.M.Sc Statistics & Data Science — Vishwakarma University, Pune  
> **Author:** Siddharth Gaikwad (31252434) &nbsp;|&nbsp; **Mentor:** Dr. Raeesa Bashir

---

## 🧾 Overview

An end-to-end AI pipeline that automatically **extracts**, **classifies**, and **structures** data from scanned corporate documents. The system combines Tesseract OCR, regex/NLTK-based NLP field extraction, and a TF-IDF + Random Forest classifier — deployed as a Flask REST API with a Streamlit frontend.

The pipeline handles **7 document types** drawn from a mixed dataset of synthetic and real-world RVL-CDIP scans:

`invoice` · `contract` · `report` · `email` · `memo` · `receipt` · `purchase_order`

---

## 📊 Results at a Glance

| Metric | Score |
|---|---|
| Overall Accuracy | **0.75** |
| Precision (weighted) | 0.86 |
| Recall (weighted) | 0.75 |
| F1-Score (weighted) | **0.74** |
| CV Mean Accuracy (4-fold) | 0.75 ± 0.06 |
| Synthetic docs F1 | 1.00 |
| RVL-CDIP (real scans) F1 | 0.40 |
| OCR A/B winner | PSM 3 (134 vs 19 words) |
| Classifier A/B winner | Random Forest (0.75 vs NB 0.67) |

---

## 🗂️ Project Structure

```
OCR_Document_Processing/
│
├── main_pipeline.py           # End-to-end orchestrator — run this
│
├── src/
│   ├── config.py              # Paths, constants, doc-type list
│   ├── data_collection.py     # Step 1: random sampling + synthetic generation
│   ├── ocr_engine.py          # Step 3: Tesseract OCR + A/B PSM test
│   ├── nlp_extractor.py       # Step 3: regex field extraction + OCR correction
│   ├── model_trainer.py       # Step 4: TF-IDF + RF + LayoutLM + A/B test
│   ├── evaluator.py           # Step 5: Accuracy, F1, confusion matrix
│   └── report_generator.py   # Auto-generate markdown project report
│
├── deployment/
│   └── backend_api.py         # Flask REST API  (/health  /process)
│
├── frontend/
│   └── streamlit_app.py       # Streamlit drag-and-drop UI
│
├── data/
│   └── raw/
│       ├── invoice/           # ← add your invoice images here
│       ├── contract/          # ← add your contract images here
│       ├── report/
│       ├── email/
│       ├── memo/
│       ├── receipt/
│       └── purchase_order/
│
├── models/                    # Saved classifier.pkl
├── reports/                   # confusion_matrix.png · PROJECT_REPORT.md
├── outputs/                   # Pipeline artefacts
├── tests/
│   ├── test_scenarios.py      # pytest suite + scenario generator
│   └── test_scenarios.csv     # Auto-generated test scenario template
│
└── requirements.txt
```

---

## ⚙️ Installation

**Prerequisites:** Python 3.9+, [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

```bash
# 1. Clone
git clone https://github.com/Siddharth-Gaikwad28/TCS-ION-Document-Processing-Using-OCR-and-Computer-vVsion.git
cd TCS-ION-Document-Processing-Using-OCR-and-Computer-vVsion

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Windows only) Set Tesseract path in src/config.py
#    TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

---

## 🚀 Quick Start

### Run the full pipeline
```bash
python main_pipeline.py
```
This runs all 5 steps in order: data collection → OCR extraction → NLP field extraction → classification + evaluation → report generation.

### Start the Flask API
```bash
python deployment/backend_api.py
# API available at http://localhost:5000
```

### Launch the Streamlit UI
```bash
streamlit run frontend/streamlit_app.py
# UI available at http://localhost:8501
```

---

## 📡 API Reference

### `GET /health`
```json
{ "status": "healthy", "service": "ocr-document-processor" }
```

### `POST /process`
Upload a document image (PNG / JPG / PDF) as `multipart/form-data`.

```bash
curl -X POST http://localhost:5000/process \
     -F "file=@invoice_scan.png"
```

**Response:**
```json
{
  "status": "success",
  "document_type": "invoice",
  "extracted_text": "INVOICE\nInvoice #: INV-4821 ...",
  "structured_fields": {
    "invoice_number": "INV-4821",
    "date": "2025-03-15",
    "amount": "1500.00",
    "vendor_name": "ABC Corp",
    "email": null,
    "total": "1500.00"
  }
}
```

---

## 🧠 Pipeline — How It Works

```
        ┌─────────────────────────────────────────────────────┐
        │              main_pipeline.py                        │
        └──────────────────────┬──────────────────────────────┘
                               │
   ┌───────────────────────────▼──────────────────────────────┐
   │  STEP 1 — Data Collection  (src/data_collection.py)      │
   │  • Randomly samples images from data/raw/<type>/          │
   │  • Auto-fills missing types with synthetic PIL images     │
   │  • Stratified across all 7 document types                 │
   └───────────────────────────┬──────────────────────────────┘
                               │
   ┌───────────────────────────▼──────────────────────────────┐
   │  STEP 3 — OCR Extraction  (src/ocr_engine.py)            │
   │  • Tesseract OCR (A/B test: PSM 6 vs PSM 3)              │
   │  • Greyscale conversion + OCR error correction            │
   └──────────────────┬────────────────────┬──────────────────┘
                      │                    │
   ┌──────────────────▼────┐    ┌──────────▼───────────────────┐
   │  NLP Extraction        │    │  Classification               │
   │  (nlp_extractor.py)   │    │  (model_trainer.py)           │
   │  • Regex patterns      │    │  • TF-IDF vectorisation       │
   │  • NLTK POS fallback   │    │  • Random Forest (200 trees)  │
   │  • 13 field types      │    │  • A/B vs Naive Bayes         │
   └────────────────────────┘    │  • 4-fold cross-validation    │
                                  └──────────┬────────────────────┘
                                             │
   ┌──────────────────────────────────────────▼──────────────────┐
   │  STEP 5 — Evaluation + Deployment                            │
   │  • Accuracy / Precision / Recall / F1 + confusion matrix    │
   │  • Flask REST API  +  Streamlit frontend                     │
   └─────────────────────────────────────────────────────────────┘
```

---

## 📂 Adding Your Own Data

Drop your document images into the matching subfolder under `data/raw/`:

```
data/raw/
  invoice/          ← PNG / JPG / TIFF invoice scans
  contract/         ← contract document images
  report/           ← quarterly or annual report images
  email/            ← printed/scanned email images
  memo/             ← internal memo images
  receipt/          ← receipt images
  purchase_order/   ← purchase order images
```

The pipeline will **randomly sample** up to `n_per_type` images from each folder and auto-generate synthetic fill-ins for any empty folder. No code changes needed.

---

## 🧪 Tests

```bash
# Run pytest suite
pytest tests/test_scenarios.py -v

# Regenerate test scenario CSV
python tests/test_scenarios.py
```

**Test coverage includes:** NLP field extraction, OCR error correction, preprocessing resize, Flask API smoke test, and test scenario template generation.

---

## 📦 Key Dependencies

| Library | Role |
|---|---|
| `pytesseract` + `opencv-python` | OCR text extraction |
| `Pillow` | Synthetic document generation |
| `scikit-learn` | TF-IDF vectorisation + Random Forest |
| `nltk` | POS-tag NER fallback |
| `flask` + `werkzeug` | REST API backend |
| `streamlit` | Interactive frontend |
| `pandas` + `numpy` | Data handling |
| `matplotlib` + `seaborn` | Confusion matrix plotting |

Optional (LayoutLM fine-tuning):
```bash
pip install torch transformers
```

---

## 📈 Key Findings

- **PSM 3 vs PSM 6:** PSM 3 extracted 7× more words on real RVL-CDIP scans (134 vs 19), making it the preferred mode for mixed-layout documents.
- **Random Forest vs Naive Bayes:** RF achieved 0.75 accuracy vs NB's 0.67 on the same dataset split.
- **Synthetic vs Real Gap:** The model scored F1 = 1.00 on clean synthetic docs but F1 = 0.40 on real RVL-CDIP scans — primarily due to OCR noise disrupting TF-IDF features.
- **Improvement roadmap:** More real-document training samples, character n-gram TF-IDF (`analyzer='char_wb'`), and LayoutLM fine-tuning are identified as the highest-impact next steps.

---

## 🔭 Future Work

- [ ] Expand RVL-CDIP training samples to 80–100 per class
- [ ] Character n-gram TF-IDF for OCR-robust features
- [ ] LayoutLM fine-tuning for multimodal document understanding
- [ ] Active learning loop — route low-confidence predictions to human review
- [ ] PDF multi-page support
- [ ] Docker containerisation for one-command deployment

---

## 📜 Dataset & Ethics

| Source | Count | License |
|---|---|---|
| Synthetic (PIL-generated) | 35 | N/A — no real data used |
| RVL-CDIP (Harley et al., 2015) | 25 | Non-commercial research |

All synthetic documents use procedurally generated dummy values. No personally identifiable information was used or stored at any stage.

---

## 📚 Citation

If you use this work, please cite:

```
Gaikwad, S. (2025). AI-Powered Document Processing System Using Vision Intelligence
and Machine Learning. TCS iON AIP 135 Curriculum Project, F.Y.M.Sc Statistics &
Data Science, Vishwakarma University, Pune.
```

---

## 🤝 Acknowledgements

- **TCS iON** — Industry project framework (AIP 135)
- **Harley et al. (2015)** — RVL-CDIP dataset
- **Google Tesseract OCR** — Open-source OCR engine

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Flask-REST%20API-black?logo=flask" />
  <img src="https://img.shields.io/badge/Streamlit-Frontend-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Tesseract-OCR-green" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/TCS%20iON-AIP%20135-blueviolet" />
</p>
