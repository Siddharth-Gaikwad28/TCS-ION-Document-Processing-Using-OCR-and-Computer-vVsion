"""
data_collection.py  —  Step 1
==============================
Sources (in priority order):
  1. Local RVL-CDIP dataset (Kaggle download, placed in data/raw/)
  2. Synthetic generator   →  PIL-drawn images that mimic each doc type

Ethical note:
  RVL-CDIP is a publicly released research dataset (Harley et al., 2015)
  licensed for non-commercial academic use. No personal or private data
  is collected. All synthetic documents contain randomly generated dummy
  values and do not represent real individuals or organisations.
"""

import os
import random
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from src.config import Config


# ---------------------------------------------------------------------------
# Synthetic generator (fallback / augmentation)
# ---------------------------------------------------------------------------

_TEMPLATES = {
    'invoice': [
        ('INVOICE',          'black'),
        ('Invoice #: INV-{rnd4}', 'black'),
        ('Date: 2025-0{d1}-{d2}', 'black'),
        ('Amount: ${amt}.00',     'black'),
        ('Vendor: ABC Corp',      'black'),
        ('Description: Consulting Services', 'black'),
    ],
    'contract': [
        ('SERVICE CONTRACT',      'black'),
        ('Contract ID: CTR-{rnd4}','black'),
        ('Party A: XYZ Ltd',      'black'),
        ('Party B: ABC Corp',     'black'),
        ('Effective: 2025-0{d1}-01','black'),
        ('Terms: 12 months',      'black'),
    ],
    'report': [
        ('QUARTERLY REPORT',      'black'),
        ('Report ID: RPT-{rnd4}', 'black'),
        ('Dept: Finance',         'black'),
        ('Period: Q1 2025',       'black'),
        ('Summary: Targets exceeded by 12%', 'black'),
    ],
    'email': [
        ('FROM: alice@example.com','black'),
        ('TO:   bob@example.com',  'black'),
        ('DATE: 2025-0{d1}-{d2}', 'black'),
        ('SUBJECT: Project Update','black'),
        ('Body: Please find the attached report for review.', 'black'),
    ],
    'memo': [
        ('INTERNAL MEMO',         'black'),
        ('TO:   All Staff',       'black'),
        ('FROM: Management',      'black'),
        ('DATE: 2025-0{d1}-{d2}','black'),
        ('RE:   Policy Update',   'black'),
        ('Action required by Friday.', 'black'),
    ],
    'receipt': [
        ('RECEIPT',               'black'),
        ('Receipt #: RCP-{rnd4}', 'black'),
        ('Date: 2025-0{d1}-{d2}','black'),
        ('Item: Office Supplies', 'black'),
        ('Qty: {d1}   Unit: $50', 'black'),
        ('Total: ${amt}.00',      'black'),
    ],
    'purchase_order': [
        ('PURCHASE ORDER',        'black'),
        ('PO #: PO-{rnd4}',      'black'),
        ('Buyer: XYZ Ltd',        'black'),
        ('Supplier: ABC Corp',    'black'),
        ('Date: 2025-0{d1}-{d2}','black'),
        ('Total: ${amt}.00',      'black'),
    ],
}

def _fmt(template: str) -> str:
    return (template
            .replace('{rnd4}', str(random.randint(1000, 9999)))
            .replace('{d1}',   str(random.randint(1, 9)))
            .replace('{d2}',   str(random.randint(10, 28)))
            .replace('{amt}',  str(random.randint(100, 9999))))


def generate_synthetic_document(doc_type: str, output_path: str) -> str:
    img  = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font_bold = ImageFont.truetype('arialbd.ttf', 24)
        font      = ImageFont.truetype('arial.ttf',   20)
    except Exception:
        font_bold = font = ImageFont.load_default()

    rows = _TEMPLATES.get(doc_type, _TEMPLATES['report'])
    for i, (text, color) in enumerate(rows):
        y = 50 + i * 50
        f = font_bold if i == 0 else font
        draw.text((50, y), _fmt(text), fill=color, font=f)

    # Simulate scanner noise
    pixels = img.load()
    for _ in range(600):
        x, y = random.randint(0, 799), random.randint(0, 999)
        pixels[x, y] = (random.randint(180, 220),) * 3

    img.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# RVL-CDIP loader  (real data — Step 1 primary source)
# ---------------------------------------------------------------------------

RVLCDIP_LABEL_MAP = {
    'invoice':        'invoice',
    'memo':           'memo',
    'email':          'email',
    'report':         'report',
    'news_article':   'report',
    'scientific_report': 'report',
    'letter':         'contract',
    'form':           'invoice',
    'advertisement':  'report',
    'scientific':     'report',
    'news':           'report',
    'specification':  'purchase_order',
    'file':           'contract',
    'budget':         'invoice',
    'handwritten':    'memo',
    'presentation':   'report',
    'questionnaire':  'receipt',
    'resume':         'memo',
}

RELEVANT_FOLDERS = set(RVLCDIP_LABEL_MAP)


def load_rvlcdip_dataset(n_samples: int = 25, save_dir: str = None) -> pd.DataFrame:
    """
    Load real scanned docs from local RVL-CDIP dataset folder.
    Falls back to empty DataFrame if unavailable.

    Ethical note: RVL-CDIP (Harley et al., 2015) is publicly released
    for non-commercial research. Source: https://huggingface.co/datasets/rvl_cdip
    """
    save_dir = save_dir or Config.RAW_DIR
    os.makedirs(save_dir, exist_ok=True)
    metadata = []

    try:
        dataset_path = r"C:\Users\pc\Desktop\TCS_AI\OCR_Document_Processing_ver2\data\raw"
        print(f"Using local dataset at: {dataset_path}")

        files_by_label = {}
        for folder_name in sorted(RELEVANT_FOLDERS):
            folder_path = os.path.join(dataset_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            label = RVLCDIP_LABEL_MAP.get(folder_name, 'report')
            for fname in sorted(os.listdir(folder_path)):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    continue
                files_by_label.setdefault(label, []).append(os.path.join(folder_path, fname))

        count = 0
        index_by_label = {label: 0 for label in files_by_label}
        ordered_labels = list(files_by_label.keys())

        while count < n_samples:
            added = False
            for label in ordered_labels:
                idx = index_by_label[label]
                label_files = files_by_label[label]
                if idx >= len(label_files):
                    continue

                src_path = label_files[idx]
                out_name = f'rvlcdip_{count}_{label}.png'
                out_path = os.path.join(save_dir, out_name)

                Image.open(src_path).convert('RGB').save(out_path)
                metadata.append({
                    'file': out_name,
                    'type': label,
                    'path': out_path,
                    'source': 'rvl_cdip'
                })
                index_by_label[label] += 1
                count += 1
                added = True
                if count >= n_samples:
                    break

            if not added:
                break

        print(f'Loaded {count} real RVL-CDIP documents from local path.')

    except Exception as e:
        print(f'RVL-CDIP unavailable ({e}). Using synthetic fallback.')

    return pd.DataFrame(metadata)


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

class DataCollector:
    DOC_TYPES = Config.DOC_TYPES

    def collect_sample_dataset(self, n_per_type: int = 5,
                                use_real: bool = True,
                                n_real_samples: int = 25) -> pd.DataFrame:
        """
        Collect dataset:
          - Tries local RVL-CDIP first if use_real=True
          - Always supplements with synthetic docs for each type
        """
        Config.init_dirs()
        metadata = []

        # ── Real data ──────────────────────────────────────────────
        if use_real:
            real_df = load_rvlcdip_dataset(n_samples=n_real_samples)
            if not real_df.empty:
                metadata.extend(real_df.to_dict('records'))
                print(f'Real label breakdown: {real_df["type"].value_counts().to_dict()}')

        # ── Synthetic (all 7 types) ────────────────────────────────
        for doc_type in self.DOC_TYPES:
            for i in range(n_per_type):
                fname    = f'{doc_type}_{i+1}.png'
                out_path = os.path.join(Config.RAW_DIR, fname)
                generate_synthetic_document(doc_type, out_path)
                metadata.append({'file': fname, 'type': doc_type,
                                 'path': out_path, 'source': 'synthetic'})

        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(Config.DATA_DIR, 'metadata.csv'), index=False)
        print(f'Dataset ready: {len(df)} documents  |  types: {df["type"].unique().tolist()}')
        if 'source' in df.columns:
            print(f'Source breakdown: {df["source"].value_counts().to_dict()}')
        return df
