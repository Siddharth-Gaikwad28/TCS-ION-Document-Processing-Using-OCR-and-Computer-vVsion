"""
ocr_engine.py  —  Step 3
===========================
Includes:
  - Tesseract OCR with configurable PSM
  - OCR error correction via nlp_extractor.correct_ocr_errors()
  - A/B test method: compare PSM 6 vs PSM 3 (Step 4 fix)
"""

import re
import cv2
import pytesseract
from src.config import Config
from src.nlp_extractor import correct_ocr_errors


class OCREngine:
    def __init__(self, psm: int = None):
        if Config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
        self.psm = psm or Config.OCR_PSM

    def _build_config(self, psm: int) -> str:
        return f'--psm {psm} --oem 3'

    def extract_text(self, image_path: str, psm: int = None) -> str:
        """Extract and normalise text from a preprocessed image."""
        psm = psm or self.psm
        img = cv2.imread(image_path)
        if img is None:
            return ''
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw  = pytesseract.image_to_string(gray,
                                           lang=Config.OCR_LANG,
                                           config=self._build_config(psm))
        return correct_ocr_errors(raw)

    # ── A/B Test: PSM 6 vs PSM 3 (Step 4 fix) ─────────────────────────────

    def ab_test_psm(self, image_path: str,
                    psm_a: int = 6, psm_b: int = 3) -> dict:
        """
        Compare two Tesseract PSM modes on the same image.
        Returns both extracted texts + word counts for comparison.

        PSM 6 — Assume uniform block of text (default, good for typed docs)
        PSM 3 — Fully automatic page segmentation (better for mixed layouts)
        """
        text_a = self.extract_text(image_path, psm=psm_a)
        text_b = self.extract_text(image_path, psm=psm_b)

        words_a = len(text_a.split())
        words_b = len(text_b.split())

        # Simple heuristic: more words = better OCR coverage
        winner = f'PSM {psm_a}' if words_a >= words_b else f'PSM {psm_b}'

        result = {
            f'PSM_{psm_a}_text':       text_a,
            f'PSM_{psm_b}_text':       text_b,
            f'PSM_{psm_a}_word_count': words_a,
            f'PSM_{psm_b}_word_count': words_b,
            'winner':                  winner,
            'recommendation':          f'{winner} extracted more text — use for this layout.'
        }
        print(f'\n[A/B Test] PSM {psm_a}: {words_a} words  |  PSM {psm_b}: {words_b} words  →  Winner: {winner}')
        return result

    @staticmethod
    def tokenize(text: str) -> list:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
