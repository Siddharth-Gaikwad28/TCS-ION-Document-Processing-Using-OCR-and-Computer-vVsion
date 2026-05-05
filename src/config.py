import os

class Config:
    BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR      = os.path.join(BASE_DIR, 'data')
    RAW_DIR       = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    MODEL_DIR     = os.path.join(BASE_DIR, 'models')
    OUTPUT_DIR    = os.path.join(BASE_DIR, 'outputs')
    REPORT_DIR    = os.path.join(BASE_DIR, 'reports')
    TEST_DIR      = os.path.join(BASE_DIR, 'tests')
    DEPLOY_DIR    = os.path.join(BASE_DIR, 'deployment')
    FRONTEND_DIR  = os.path.join(BASE_DIR, 'frontend')
    SRC_DIR       = os.path.join(BASE_DIR, 'src')

    # Tesseract — Windows example:
    # TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_CMD = None
    OCR_LANG      = 'eng'
    OCR_PSM       = 6
    RANDOM_STATE  = 42
    TEST_SIZE     = 0.2

    # Expanded document types (Step 1 fix)
    DOC_TYPES = ['invoice', 'contract', 'report', 'email', 'memo', 'receipt', 'purchase_order']

    @classmethod
    def init_dirs(cls):
        for d in [cls.RAW_DIR, cls.PROCESSED_DIR, cls.MODEL_DIR,
                  cls.OUTPUT_DIR, cls.REPORT_DIR, cls.TEST_DIR,
                  cls.DEPLOY_DIR, cls.FRONTEND_DIR, cls.SRC_DIR]:
            os.makedirs(d, exist_ok=True)
        print('Directories ready:', cls.BASE_DIR)
