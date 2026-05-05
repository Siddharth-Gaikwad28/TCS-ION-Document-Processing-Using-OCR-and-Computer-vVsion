"""
evaluator.py - Step 4
=====================
Accuracy / Precision / Recall / F1 / Confusion Matrix
Benchmarks on both synthetic and real (RVL-CDIP) samples.
"""

import os
import re

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import Config


class Evaluator:
    @staticmethod
    def _resolve_eval_path(path: str) -> str:
        base = os.path.splitext(os.path.basename(path))[0]
        processed_path = os.path.join(Config.PROCESSED_DIR, base + ".png")
        return processed_path if os.path.exists(processed_path) else path

    @staticmethod
    def evaluate_ocr_accuracy(ground_truth: str, extracted: str) -> float:
        gt = re.sub(r"\s+", "", ground_truth.lower())
        ex = re.sub(r"\s+", "", extracted.lower())
        if not gt:
            return 0.0
        return sum(a == b for a, b in zip(gt, ex)) / max(len(gt), len(ex))

    @staticmethod
    def evaluate_classification(y_true: list, y_pred: list) -> tuple:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
        print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
        return metrics, cm

    @staticmethod
    def plot_confusion_matrix(cm, labels: list, save_path: str = None, title: str = "Confusion Matrix"):
        save_path = save_path or os.path.join(Config.REPORT_DIR, "confusion_matrix.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved -> {save_path}")

    def benchmark_sources(self, meta_df, ocr, clf) -> dict:
        """
        Separately evaluate performance on real (rvl_cdip) vs synthetic docs.
        Uses the preprocessed images when available so this matches the main evaluation path.
        """
        results = {}
        sources = meta_df["source"].unique() if "source" in meta_df.columns else ["all"]
        for source in sources:
            subset = meta_df[meta_df["source"] == source] if source != "all" else meta_df
            if subset.empty:
                continue

            y_true = subset["type"].tolist()
            y_pred = []
            for path in subset["path"]:
                eval_path = self._resolve_eval_path(path)
                text = ocr.extract_text(eval_path)
                y_pred.append(clf.predict(text)[0] if text else "")

            metrics, _ = self.evaluate_classification(y_true, y_pred)
            results[source] = metrics
            print(f'\n[Benchmark - {source}] F1={metrics["f1_score"]:.2f}  Acc={metrics["accuracy"]:.2f}')
        return results
