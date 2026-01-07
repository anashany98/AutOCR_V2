try:
    import paddleocr
    print(f"PaddleOCR Version: {paddleocr.__version__}")
    print(f"Has PPStructure? {'PPStructure' in dir(paddleocr)}")
    print(f"Has PaddleOCR? {'PaddleOCR' in dir(paddleocr)}")
    
    from paddleocr import PaddleOCR
    import inspect
    print("PaddleOCR args:", inspect.signature(PaddleOCR.__init__))
except ImportError as e:
    print(f"Error: {e}")
