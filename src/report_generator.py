import os

from src.config import Config


class ReportGenerator:
    @staticmethod
    def _fmt_metric(value) -> str:
        return f'{value:.2f}' if isinstance(value, (int, float)) else str(value)

    def generate_project_report(self, metrics: dict, ab_results: dict = None) -> str:
        ab_section = ''
        if ab_results:
            ab_section = f"""
## 5. A/B Test Results
### OCR: PSM 6 vs PSM 3
| Mode  | Word Count | Winner |
|-------|-----------|--------|
| PSM 6 | {ab_results.get('ocr', {}).get('PSM_6_word_count', 'N/A')} | {'yes' if 'PSM 6' in str(ab_results.get('ocr', {}).get('winner', '')) else ''} |
| PSM 3 | {ab_results.get('ocr', {}).get('PSM_3_word_count', 'N/A')} | {'yes' if 'PSM 3' in str(ab_results.get('ocr', {}).get('winner', '')) else ''} |

### Classifier: RandomForest vs NaiveBayes
| Model        | Accuracy | Winner |
|--------------|----------|--------|
| RandomForest | {ab_results.get('clf', {}).get('RandomForest', 'N/A')} | {'yes' if ab_results.get('clf', {}).get('winner') == 'RandomForest' else ''} |
| NaiveBayes   | {ab_results.get('clf', {}).get('NaiveBayes',   'N/A')} | {'yes' if ab_results.get('clf', {}).get('winner') == 'NaiveBayes'   else ''} |
"""

        report = f"""# Project Report: Document Processing & Data Extraction (OCR Pipeline)
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
| Accuracy  | {self._fmt_metric(metrics.get('accuracy',  0))} |
| Precision | {self._fmt_metric(metrics.get('precision', 0))} |
| Recall    | {self._fmt_metric(metrics.get('recall',    0))} |
| F1-Score  | {self._fmt_metric(metrics.get('f1_score',  0))} |

## 4. Challenges & Solutions
| Challenge              | Solution                                              |
|------------------------|-------------------------------------------------------|
| OCR character errors   | `correct_ocr_errors()` dict + regex post-fixes        |
| Single dataset bias    | RVL-CDIP real docs + synthetic augmentation           |
| Layout variations      | LayoutLM fine-tuning on tokenized OCR output          |
| Low-confidence outputs | A/B PSM test selects optimal Tesseract page-seg mode  |
{ab_section}
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
"""
        path = os.path.join(Config.REPORT_DIR, 'PROJECT_REPORT.md')
        os.makedirs(Config.REPORT_DIR, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'Report saved -> {path}')
        return path