# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['document_processor.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Если есть модели или конфиги
        # ('models/', 'models'),
        # ('config.json', '.'),
    ],
    hiddenimports=[
        # Явное указание модулей для OCR
        'easyocr',
        'easyocr.model',
        'easyocr.detection',
        'easyocr.recognition',
        'cv2',
        'numpy',
        'PIL',
        'pytesseract',
        'pdf2image',
        'camelot',
        'rapidocr_onnxruntime',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DocumentOCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # True для консольной программы
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',  # Иконка (опционально)
)
