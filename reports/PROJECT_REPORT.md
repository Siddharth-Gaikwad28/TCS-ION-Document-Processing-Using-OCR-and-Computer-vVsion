# Project Report: Document Processing & Data Extraction (OCR Pipeline)
## TCS iON Industry Project

## 1. Architecture
- **Frontend**: Streamlit UI (`frontend/streamlit_app.py`) — calls Flask API or runs locally
- **Backend**: Flask REST API (`deployment/backend_api.py`) — `/health`, `/process`
- **OCR**: Tesseract (PSM 6 default) with A/B PSM test (`src/ocr_engine.py`)
- **NLP**: Regex field extractor + NLTK POS fallback (`src/nlp_extractor.py`)
- **ML**: TF-IDF + RandomForest classifier; LayoutLM fine-tune available (`src/model_trainer.py`)
- **Deployment**: Local Flask server; threading-based in-notebook execution

## 2. Pipeline Steps
1. **Data Collection** — random sampling across document types from dataset
2. **OCR Extraction** — Tesseract extracts raw text; A/B test selects best PSM mode
3. **NLP Field Extraction** — regex + NLTK extracts structured fields (invoice #, dates, amounts)
4. **Classification** — TF-IDF vectorisation → RandomForest predicts document type
5. **Evaluation** — Accuracy, Precision, Recall, F1 + Confusion Matrix
6. **Deployment** — Flask REST API + Streamlit UI

## 3. Classification Metrics
| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.75 |
| Precision | 0.86 |
| Recall    | 0.75 |
| F1-Score  | 0.74 |

## 4. Challenges & Solutions
| Challenge              | Solution                                              |
|------------------------|-------------------------------------------------------|
| OCR character errors   | `correct_ocr_errors()` dict + regex post-fixes        |
| Single dataset bias    | RVL-CDIP real docs + synthetic augmentation           |
| Layout variations      | LayoutLM fine-tuning on tokenized OCR output          |
| Low-confidence outputs | A/B PSM test selects optimal Tesseract page-seg mode  |

## 5. A/B Test Results
### OCR: PSM 6 vs PSM 3
| Mode  | Word Count | Winner |
|-------|-----------|--------|
| PSM 6 | 19 |  |
| PSM 3 | 134 | yes |

### Classifier: RandomForest vs NaiveBayes
| Model        | Accuracy | Winner |
|--------------|----------|--------|
| RandomForest | 0.75 | yes |
| NaiveBayes   | 0.6666666666666666 |  |

## 6. Data Sources
- **RVL-CDIP** (Harley et al., 2015): 400k real scanned corporate docs.
  Source: https://huggingface.co/datasets/rvl_cdip — non-commercial research license.
- **Synthetic**: PIL-generated dummy invoices, contracts, reports, emails, memos,
  receipts, purchase orders. No real personal data used.

## 7. Deliverables Checklist
- [x] Python source modules (`src/`)
- [x] Test Scenario Template CSV (`tests/test_scenarios.csv`)
- [x] Evaluation report + confusion matrix (`reports/`)
- [x] Flask backend API (`deployment/`)
- [x] Streamlit frontend (`frontend/`)
- [x] A/B test results (OCR PSM + classifier)
- [ ] Video demo — record manually using Streamlit + Flask running together
