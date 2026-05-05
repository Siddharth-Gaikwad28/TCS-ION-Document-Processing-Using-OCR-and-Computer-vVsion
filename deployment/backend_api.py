"""
backend_api.py  —  Step 4/5
==============================
Flask REST API for OCR document processing.

Run standalone:
    python deployment/backend_api.py

Run from inside a Jupyter notebook (threading):
    from deployment.backend_api import start_in_thread
    start_in_thread()
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from src.config import Config
from src.ocr_engine import OCREngine
from src.nlp_extractor import NLPExtractor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ocr = OCREngine()
nlp = NLPExtractor()

Config.init_dirs()

ALLOWED = {'png', 'jpg', 'jpeg', 'pdf'}

def _allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'ocr-document-processor'})


@app.route('/process', methods=['POST'])
def process_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file in request'}), 400
    file = request.files['file']
    if not file.filename or not _allowed(file.filename):
        return jsonify({'error': f'Unsupported file. Allowed: {ALLOWED}'}), 400

    filename  = secure_filename(file.filename)
    temp_path = os.path.join(Config.UPLOAD_DIR, filename)
    file.save(temp_path)

    try:
        # Extract text directly from uploaded file — no image preprocessing
        text     = ocr.extract_text(temp_path)
        fields   = nlp.extract_fields(text)
        doc_type = nlp.classify_document(text)

        # Use trained ML classifier if available
        clf_path = os.path.join(Config.MODEL_DIR, 'classifier.pkl')
        if os.path.exists(clf_path):
            from src.model_trainer import DocumentClassifier
            clf = DocumentClassifier()
            clf.load(clf_path)
            doc_type, _ = clf.predict(text)

        return jsonify({
            'status':            'success',
            'document_type':     doc_type,
            'extracted_text':    text,
            'structured_fields': fields,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def start_in_thread(port: int = 5000):
    """
    Call this from a Jupyter notebook cell to start Flask in background:

        from deployment.backend_api import start_in_thread
        start_in_thread()
    """
    import threading
    t = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=port, use_reloader=False, debug=False),
        daemon=True
    )
    t.start()
    print(f'Flask API running at http://127.0.0.1:{port}')
    print('  GET  /health')
    print('  POST /process   (form-data: file=<image/pdf>)')
    return t


if __name__ == '__main__':
    print('Starting OCR API…')
    app.run(host='0.0.0.0', port=5000, debug=True)