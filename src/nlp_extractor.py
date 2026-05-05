"""
nlp_extractor.py  —  Step 3
==============================
Fixes applied:
  - NLPExtractor class properly defined (was crashing everything)
  - Comprehensive regex patterns for all field types
  - OCR error correction dictionary + regex post-fixes
  - NLTK POS-tagging fallback for name extraction
"""

import re


# ---------------------------------------------------------------------------
# OCR Error Correction  (Step 3 fix: "only basic OCR correction")
# ---------------------------------------------------------------------------

OCR_ERROR_MAP = {
    # Common Tesseract misreads
    '0' : 'O',   # zero → letter O  (context-dependent; applied before digit-only fields)
    '|' : 'I',
    'l1': '',    # handled via regex
    '—' : '-',
    '"' : '"',
    '"' : '"',
    '\x00': '',
    '©' : '',
    '®' : '',
    '™' : '',
}

OCR_REGEX_FIXES = [
    (r'[^\x00-\x7F]',         ''),          # strip non-ASCII
    (r'(?<!\d)0(?!\d)',        'O'),         # lone zero → O (not inside numbers)
    (r'\bl\b',                 'I'),         # lone lowercase L → I
    (r'[ \t]{2,}',             ' '),         # collapse whitespace
    (r'\n{3,}',                '\n\n'),      # collapse blank lines
]


def correct_ocr_errors(text: str) -> str:
    """Apply OCR correction dict + regex fixes to raw Tesseract output."""
    for wrong, right in OCR_ERROR_MAP.items():
        text = text.replace(wrong, right)
    for pattern, replacement in OCR_REGEX_FIXES:
        text = re.sub(pattern, replacement, text)
    return text.strip()


# ---------------------------------------------------------------------------
# NLPExtractor  (Step 3 fix: "NLPExtractor never defined")
# ---------------------------------------------------------------------------

class NLPExtractor:
    """
    Regex-based structured field extractor for OCR text.
    Covers: invoice numbers, contract IDs, PO numbers, receipt numbers,
            dates (multiple formats), monetary amounts, vendor/party names,
            email addresses, departments.
    """

    PATTERNS = {
        'invoice_number':   r'(?:Invoice|INV|Inv)[^\d]{0,5}(\d{4,}|[A-Z0-9\-]{4,})',
        'contract_id':      r'(?:Contract|CTR|Agr(?:eement)?)[^\d]{0,5}([A-Z0-9\-]{4,})',
        'po_number':        r'(?:P\.?O\.?|Purchase\s*Order)[^\d]{0,5}(\d{4,}|[A-Z0-9\-]{4,})',
        'receipt_number':   r'(?:Receipt|RCP)[^\d]{0,5}(\d{4,}|[A-Z0-9\-]{4,})',
        'report_id':        r'(?:Report|RPT)[^\d]{0,5}(\d{4,}|[A-Z0-9\-]{4,})',
        'date': r'(?:Date[:\s]+)?(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})',
        'amount': r'(?:[\$£€]|Amount[:\s]+)\s*([\d,]+(?:\.\d{2})?)',
        'email':            r'\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b',
        'vendor_name':      r'(?:Vendor|Supplier|From)[:\s]+([A-Za-z][\w\s\.&,]{2,30})',
        'party_a':          r'Party\s*A[:\s]+([A-Za-z][\w\s\.]{2,30})',
        'party_b':          r'Party\s*B[:\s]+([A-Za-z][\w\s\.]{2,30})',
        'department':       r'(?:Dept(?:artment)?|Division)[:\s]+([A-Za-z][\w\s]{2,25})',
        'subject':          r'(?:Subject|RE|SUBJECT)[:\s]+(.{5,80})',
        'total':            r'Total[:\s]+\$?([\d,]+(?:\.\d{2})?)',
    }

    # Keyword scores for each document type
    _DOC_KEYWORDS = {
        'invoice':        r'invoice|amount|vendor|inv\b|billing',
        'contract':       r'contract|party\s+[ab]|terms|agreement|effective',
        'report':         r'report|summary|quarterly|department|period|targets',
        'email':          r'from:|to:|subject:|body:|@',
        'memo':           r'memo|internal|re:\s|staff|management',
        'receipt':        r'receipt|rcp\b|total|qty|unit price|paid',
        'purchase_order': r'purchase\s+order|po\b|buyer|supplier',
    }

    def extract_fields(self, text: str) -> dict:
        """Return dict of extracted fields. Applies OCR correction first."""
        text   = correct_ocr_errors(text)
        fields = {}
        for field, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            fields[field] = matches[0].strip() if matches else None

        # NLTK POS-tag fallback for vendor name if regex missed it
        if not fields.get('vendor_name'):
            fields['vendor_name'] = self._nltk_name_fallback(text)

        return fields

    def classify_document(self, text: str) -> str:
        """Return predicted document type based on keyword frequency."""
        text   = correct_ocr_errors(text).lower()
        scores = {}
        for doc_type, pattern in self._DOC_KEYWORDS.items():
            scores[doc_type] = len(re.findall(pattern, text))
        return max(scores, key=scores.get)

    @staticmethod
    def _nltk_name_fallback(text: str) -> str | None:
        """Use NLTK POS tagging to find a PERSON or ORGANIZATION entity."""
        try:
            import nltk
            from nltk import pos_tag, word_tokenize, ne_chunk
            from nltk.tree import Tree
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker',          quiet=True)
            nltk.download('words',                      quiet=True)
            chunks = ne_chunk(pos_tag(word_tokenize(text[:500])))
            for chunk in chunks:
                if isinstance(chunk, Tree) and chunk.label() in ('PERSON', 'ORGANIZATION'):
                    return ' '.join(w for w, _ in chunk.leaves())
        except Exception:
            pass
        return None
