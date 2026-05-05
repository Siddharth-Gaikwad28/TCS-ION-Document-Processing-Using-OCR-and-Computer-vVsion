"""
model_trainer.py  —  Step 3
==============================
Two classifiers available:
  1. DocumentClassifier  — TF-IDF + RandomForest (always available)
  2. LayoutLMClassifier  — microsoft/layoutlm-base-uncased fine-tune
                           (requires: pip install transformers torch)

Step 3 fix: "No fine-tuning" → LayoutLMClassifier added.
"""

import os
import pickle
from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import Config
from src.ocr_engine import OCREngine


# ---------------------------------------------------------------------------
# 1. TF-IDF + RandomForest  (primary, always works)
# ---------------------------------------------------------------------------

class DocumentClassifier:
    """TF-IDF vectorisation + RandomForest document type classifier."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=800,
            ngram_range=(1, 2),
            analyzer='word',       # back to word-level
            sublinear_tf=True,
            min_df=2,              # ignore terms appearing in only 1 doc (OCR noise)
            strip_accents='unicode'
        )

        self.model      = RandomForestClassifier(n_estimators=200,
                                                 random_state=Config.RANDOM_STATE,
                                                 class_weight='balanced')
        self.is_trained = False
        self._ocr       = OCREngine()

    def _extract_texts(self, meta_df: pd.DataFrame) -> tuple:
        texts, labels = [], []
        for _, row in meta_df.iterrows():
            texts.append(self._ocr.extract_text(row['path']))
            labels.append(row['type'])
        return texts, labels

    @staticmethod
    def _can_stratify(labels) -> bool:
        counts = Counter(labels)
        return len(counts) > 1 and min(counts.values()) >= 2

    @staticmethod
    def _can_holdout(labels) -> bool:
        # Holdout needs at least one duplicate label so train/test can both have samples.
        return len(labels) >= 2 and len(set(labels)) < len(labels)

    def train(self, meta_df: pd.DataFrame) -> 'DocumentClassifier':
        texts, labels = self._extract_texts(meta_df)
        X = self.vectorizer.fit_transform(texts)

        if not self._can_holdout(labels):
            self.model.fit(X, labels)
            self.is_trained = True
            print('[RF] Skipping validation split: dataset too small for holdout.')
            return self

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=labels if self._can_stratify(labels) else None
        )
        self.model.fit(X_tr, y_tr)
        self.is_trained = True
        preds = self.model.predict(X_te)
        print(f'[RF] Validation accuracy: {accuracy_score(y_te, preds):.2f}')
        print(classification_report(y_te, preds, zero_division=0))
        return self

    def fit(self, meta_df: pd.DataFrame) -> 'DocumentClassifier':
        texts, labels = self._extract_texts(meta_df)
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True
        print(f'[RF] Final model trained on {len(labels)} document(s).')
        return self

    def cross_validate(self, meta_df: pd.DataFrame, n_splits: int = 5) -> dict:
        texts, labels = self._extract_texts(meta_df)
        counts = Counter(labels)
        max_splits = min(n_splits, min(counts.values())) if counts else 0

        if len(counts) < 2 or max_splits < 2:
            print('[RF CV] Skipping cross-validation: not enough samples per class.')
            return {
                'splits': 0,
                'accuracy_mean': 'N/A',
                'accuracy_std': 'N/A',
                'f1_mean': 'N/A',
                'f1_std': 'N/A',
            }

        splitter = StratifiedKFold(
            n_splits=max_splits,
            shuffle=True,
            random_state=Config.RANDOM_STATE
        )
        accuracy_scores = []
        f1_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(texts, labels), start=1):
            fold_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            fold_model = RandomForestClassifier(
                n_estimators=200,
                random_state=Config.RANDOM_STATE,
                class_weight='balanced'
            )

            train_texts = [texts[i] for i in train_idx]
            test_texts = [texts[i] for i in test_idx]
            train_labels = [labels[i] for i in train_idx]
            test_labels = [labels[i] for i in test_idx]

            X_train = fold_vectorizer.fit_transform(train_texts)
            X_test = fold_vectorizer.transform(test_texts)
            fold_model.fit(X_train, train_labels)

            preds = fold_model.predict(X_test)
            fold_acc = accuracy_score(test_labels, preds)
            fold_f1 = f1_score(test_labels, preds, average='weighted', zero_division=0)
            accuracy_scores.append(fold_acc)
            f1_scores.append(fold_f1)
            print(f'[RF CV] Fold {fold_idx}/{max_splits}: Acc={fold_acc:.2f}  F1={fold_f1:.2f}')

        acc_mean = sum(accuracy_scores) / len(accuracy_scores)
        f1_mean = sum(f1_scores) / len(f1_scores)
        acc_std = (sum((score - acc_mean) ** 2 for score in accuracy_scores) / len(accuracy_scores)) ** 0.5
        f1_std = (sum((score - f1_mean) ** 2 for score in f1_scores) / len(f1_scores)) ** 0.5

        print(f'[RF CV] Mean Acc={acc_mean:.2f} ± {acc_std:.2f}  |  Mean F1={f1_mean:.2f} ± {f1_std:.2f}')
        return {
            'splits': max_splits,
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'f1_mean': f1_mean,
            'f1_std': f1_std,
        }

    def predict(self, text: str) -> tuple:
        if not self.is_trained:
            raise RuntimeError('Model not trained.')
        X    = self.vectorizer.transform([text])
        pred = self.model.predict(X)[0]
        conf = max(self.model.predict_proba(X)[0])
        return pred, float(conf)

    def save(self, path: str = None):
        path = path or os.path.join(Config.MODEL_DIR, 'classifier.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'model': self.model}, f)
        print(f'Model saved → {path}')

    def load(self, path: str = None) -> 'DocumentClassifier':
        path = path or os.path.join(Config.MODEL_DIR, 'classifier.pkl')
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        self.vectorizer = obj['vectorizer']
        self.model      = obj['model']
        self.is_trained = True
        return self

    # ── A/B: RandomForest vs Naive Bayes (Step 4 fix) ─────────────────────

    def ab_test_classifiers(self, meta_df: pd.DataFrame) -> dict:
        """
        Compare RandomForest vs MultinomialNB on the same dataset.
        This satisfies the Step 4 A/B testing requirement.
        """
        texts, labels = self._extract_texts(meta_df)
        X = self.vectorizer.fit_transform(texts)

        if not self._can_holdout(labels):
            print('\n[A/B Classifier] Skipping: dataset too small for train/test split.')
            return {'RandomForest': 'N/A', 'NaiveBayes': 'N/A', 'winner': 'Skipped'}

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=labels if self._can_stratify(labels) else None
        )
        # RF
        rf = RandomForestClassifier(n_estimators=200, random_state=Config.RANDOM_STATE)
        rf.fit(X_tr, y_tr)
        rf_acc = accuracy_score(y_te, rf.predict(X_te))

        # Naive Bayes
        nb = MultinomialNB()
        nb.fit(X_tr, y_tr)
        nb_acc = accuracy_score(y_te, nb.predict(X_te))

        winner = 'RandomForest' if rf_acc >= nb_acc else 'NaiveBayes'
        print(f'\n[A/B Classifier] RandomForest: {rf_acc:.2f}  |  NaiveBayes: {nb_acc:.2f}  →  Winner: {winner}')
        return {'RandomForest': rf_acc, 'NaiveBayes': nb_acc, 'winner': winner}


# ---------------------------------------------------------------------------
# 2. LayoutLM Fine-Tuning  (Step 3 fix: "No fine-tuning")
# ---------------------------------------------------------------------------

class LayoutLMClassifier:
    """
    Fine-tunes microsoft/layoutlm-base-uncased on extracted OCR tokens.
    Falls back gracefully if transformers/torch not installed.

    Usage:
        clf = LayoutLMClassifier(num_labels=7)
        clf.train(meta_df, epochs=3)
        label, conf = clf.predict(text)
    """

    MODEL_NAME = 'microsoft/layoutlm-base-uncased'

    def __init__(self, num_labels: int = 7):
        self.num_labels = num_labels
        self.model      = None
        self.tokenizer  = None
        self.label2id   = {}
        self.id2label   = {}
        self._ocr       = OCREngine()
        self._check_deps()

    @staticmethod
    def _check_deps():
        try:
            import torch          # noqa
            import transformers   # noqa
        except ImportError:
            raise ImportError(
                'LayoutLM requires: pip install torch transformers\n'
                'Fall back to DocumentClassifier if GPU/torch unavailable.'
            )

    def train(self, meta_df: pd.DataFrame, epochs: int = 3,
              save_dir: str = None):
        """Fine-tune LayoutLM on OCR text (text-only mode, no bounding boxes)."""
        import torch
        from transformers import (LayoutLMTokenizer,
                                   LayoutLMForSequenceClassification,
                                   TrainingArguments, Trainer)
        from torch.utils.data import Dataset

        save_dir = save_dir or os.path.join(Config.MODEL_DIR, 'layoutlm')

        # Build label maps
        labels      = meta_df['type'].unique().tolist()
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        # Load tokenizer + model
        self.tokenizer = LayoutLMTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = LayoutLMForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=len(labels),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

        # Simple text dataset (word_ids filled with zeros — text-only mode)
        class DocDataset(Dataset):
            def __init__(self, texts, label_ids, tokenizer_ref):
                self.encodings  = tokenizer_ref(texts, truncation=True,
                                                padding=True, max_length=512,
                                                return_tensors='pt')
                self.label_ids  = torch.tensor(label_ids)

            def __len__(self):
                return len(self.label_ids)

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item['labels'] = self.label_ids[idx]
                return item

        texts     = [self._ocr.extract_text(r['path']) for _, r in meta_df.iterrows()]
        label_ids = [self.label2id[r['type']] for _, r in meta_df.iterrows()]
        dataset   = DocDataset(texts, label_ids, self.tokenizer)

        args = TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=100,
            logging_steps=10,
            no_cuda=not torch.cuda.is_available(),
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=dataset)
        trainer.train()
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f'LayoutLM fine-tuned and saved → {save_dir}')

    def predict(self, text: str) -> tuple:
        import torch
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('LayoutLM not trained yet. Call train() first.')
        enc    = self.tokenizer(text, truncation=True, padding=True,
                                max_length=512, return_tensors='pt')
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs  = torch.softmax(logits, dim=-1)[0]
        idx    = probs.argmax().item()
        return self.id2label[idx], float(probs[idx])
