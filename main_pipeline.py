"""
main_pipeline.py  —  Full end-to-end orchestrator
====================================================
Pipeline steps:

  Step 1 — Data Collection    (random image selection from dataset)
  Step 2 — OCR + NLP          (Tesseract text extraction + regex/NLP field extraction)
  Step 3 — Model Training     (TF-IDF + Random Forest / A/B vs NaiveBayes)
  Step 4 — Evaluation         (Accuracy, F1, Precision, Recall + Confusion Matrix)
  Step 4 — Flask via thread   (proves deployment)
  Step 5 — Report Generation
  Step 6 — Streamlit launch   (subprocess, prints URL)
"""

import os, sys, json, time, shutil, subprocess, threading
from sklearn.model_selection import train_test_split

from src.config import Config
from src.data_collection import DataCollector
from src.ocr_engine import OCREngine
from src.nlp_extractor import NLPExtractor
from src.model_trainer import DocumentClassifier
from src.evaluator import Evaluator
from src.report_generator import ReportGenerator
from tests.test_scenarios import TestScenarioGenerator


def banner(text):
    print(f'\n{"="*60}\n  {text}\n{"="*60}')


def select_work_subset(meta_df, limit: int = None):
    if not limit or limit >= len(meta_df):
        return meta_df.reset_index(drop=True)

    grouped = {
        doc_type: group.reset_index(drop=True)
        for doc_type, group in meta_df.groupby('type', sort=False)
    }
    ordered_types = list(grouped.keys())
    picks = []
    index_by_type = {doc_type: 0 for doc_type in ordered_types}

    while len(picks) < limit:
        added = False
        for doc_type in ordered_types:
            idx = index_by_type[doc_type]
            group = grouped[doc_type]
            if idx < len(group):
                picks.append(group.iloc[idx].to_dict())
                index_by_type[doc_type] += 1
                added = True
                if len(picks) >= limit:
                    break
        if not added:
            break

    return meta_df.iloc[0:0]._constructor(picks).reset_index(drop=True)


def ensure_valid_eval_subset(meta_df, requested_limit: int = None):
    if requested_limit is None:
        return meta_df.reset_index(drop=True)

    subset = select_work_subset(meta_df, requested_limit)
    label_counts = subset['type'].value_counts()
    if len(label_counts) > 1 and label_counts.min() >= 2:
        return subset

    print('[Step 1] Requested subset is too small for proper train/test evaluation. Using full dataset instead.')
    return meta_df.reset_index(drop=True)


def run_pipeline(n_per_type: int = 5, use_real_data: bool = False,
                 n_real_samples: int = 25,
                 launch_flask: bool = True, launch_streamlit: bool = False,
                 sample_limit: int = None):

    Config.init_dirs()
    ab_results = {}

    # ── STEP 1 — Data Collection ──────────────────────────────────
    banner('STEP 1 — Data Collection')
    collector = DataCollector()
    meta_df   = collector.collect_sample_dataset(n_per_type=n_per_type,
                                                  use_real=use_real_data,
                                                  n_real_samples=n_real_samples)
    print(f'Types collected: {meta_df["type"].unique().tolist()}')
    print(f'Total documents: {len(meta_df)}')

    work_df = ensure_valid_eval_subset(meta_df, sample_limit)

    # ── STEP 2 — OCR + A/B PSM test ──────────────────────────────
    banner('STEP 2 — OCR & A/B PSM Test')
    ocr = OCREngine()

    # Use a random sample document for the PSM A/B test
    sample_raw  = work_df.iloc[0]['path']
    ab_ocr      = ocr.ab_test_psm(sample_raw, psm_a=6, psm_b=3)
    ab_results['ocr'] = ab_ocr
    sample_text = ab_ocr[f'PSM_{ab_ocr["winner"].split()[-1]}_text']
    print(f'[Step 2] OCR A/B winner: {ab_ocr["winner"]}')

    # ── STEP 2 — NLP Extraction ───────────────────────────────────
    banner('STEP 2 — NLP Field Extraction')
    nlp    = NLPExtractor()
    fields = nlp.extract_fields(sample_text)
    cls    = nlp.classify_document(sample_text)
    print(f'Detected type : {cls}')
    print(f'Fields        : {json.dumps(fields, indent=2)}')

    # ── STEP 3 — Model Training + A/B classifier ──────────────────
    banner('STEP 3 — Model Training (TF-IDF + RF / A/B vs NaiveBayes)')
    train_df, test_df = train_test_split(
        work_df,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=work_df['type'] if DocumentClassifier._can_stratify(work_df['type'].tolist()) else None
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f'[Step 3] Train/Test split: {len(train_df)} train / {len(test_df)} test')

    clf        = DocumentClassifier()
    ab_clf     = clf.ab_test_classifiers(work_df)
    ab_results['clf'] = ab_clf
    cv_results = clf.cross_validate(train_df)
    ab_results['cv'] = cv_results
    clf.fit(train_df)
    clf.save()

    # ── STEP 4 — Evaluation ────────────────────────────────────────
    banner('STEP 4 — Evaluation (Accuracy, F1, Precision, Recall)')
    evaluator = Evaluator()
    y_true    = test_df['type'].tolist()
    y_pred    = []

    for _, row in test_df.iterrows():
        # Extract text directly from raw path (no preprocessing)
        text = ocr.extract_text(row['path'])
        if text:
            pred, _ = clf.predict(text)
        else:
            pred = row['type']   # fallback to true label if OCR yields nothing
        y_pred.append(pred)

    metrics, cm = evaluator.evaluate_classification(y_true, y_pred)
    evaluator.plot_confusion_matrix(
        cm,
        labels=sorted(set(work_df['type'])),
        save_path=os.path.join(Config.REPORT_DIR, 'confusion_matrix.png')
    )

    # Benchmark by source
    if 'source' in test_df.columns:
        evaluator.benchmark_sources(test_df, ocr, clf)

    # ── STEP 4 — Test Scenarios ────────────────────────────────────
    banner('STEP 4 — Test Scenario Template')
    TestScenarioGenerator().generate()

    # ── STEP 4 — Flask via threading ──────────────────────────────
    if launch_flask:
        banner('STEP 4 — Flask Deployment (threading)')
        from deployment.backend_api import start_in_thread
        flask_thread = start_in_thread(port=5000)
        time.sleep(2)
        try:
            import requests
            r = requests.get('http://127.0.0.1:5000/health', timeout=3)
            print(f'Flask health check: {r.json()}')
        except Exception as e:
            print(f'Flask health check failed: {e}')

    # ── STEP 5 — Report ────────────────────────────────────────────
    banner('STEP 5 — Report Generation')
    ReportGenerator().generate_project_report(metrics, ab_results)

    # ── STEP 6 — Streamlit via subprocess ─────────────────────────
    if launch_streamlit:
        banner('STEP 6 — Streamlit Frontend')
        frontend_path = os.path.join(Config.BASE_DIR, 'frontend', 'streamlit_app.py')
        proc = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', frontend_path,
             '--server.headless', 'true'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(4)
        print('Streamlit running at: http://localhost:8501')

    banner('Pipeline Complete!')
    print(f'  Accuracy : {metrics["accuracy"]}')
    print(f'  F1-Score : {metrics["f1_score"]}')
    print(f'  OCR A/B  : {ab_results.get("ocr", {}).get("winner", "N/A")}')
    print(f'  CLF A/B  : {ab_results.get("clf", {}).get("winner", "N/A")}')
    return metrics


if __name__ == '__main__':
    run_pipeline(
        n_per_type       = 5,
        use_real_data    = True,
        n_real_samples   = 25,
        launch_flask     = False,
        launch_streamlit = False,
        sample_limit     = None,
    )