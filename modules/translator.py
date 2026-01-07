import logging
# Optional dependency
try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

logger = logging.getLogger(__name__)

class Translator:
    """Helper for document translation."""
    
    def __init__(self):
        self.enabled = GoogleTranslator is not None
        if not self.enabled:
            logger.warning("deep-translator not found. Translation will be unavailable.")

    def translate_text(self, text: str, target_lang: str = 'es') -> str:
        if not self.enabled:
            return "Error: Module 'deep-translator' not installed."
        
        try:
            # Check length, split if needed (Google Translate limit ~5000 chars)
            # simplistic approach for demo
            if len(text) > 4000:
                chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
                translated = []
                for chunk in chunks:
                    translated.append(GoogleTranslator(source='auto', target=target_lang).translate(chunk))
                return " ".join(translated)
            else:
                return GoogleTranslator(source='auto', target=target_lang).translate(text)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return f"Error translating: {e}"
