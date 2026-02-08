import os
import logging
from typing import Optional

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger(__name__)

class VoiceManager:
    """
    Manages local speech-to-text transcription using faster-whisper.
    """
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self.enabled = WhisperModel is not None

    def _init_model(self):
        if not self.enabled:
            logger.warning("faster-whisper no está instalado. Transcripción de voz deshabilitada.")
            return

        if self._model is None:
            try:
                logger.info(f"Cargando modelo Whisper ({self.model_size}) en {self.device}...")
                self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
                logger.info("Modelo Whisper cargado exitosamente.")
            except Exception as e:
                logger.error(f"Error al cargar modelo Whisper: {e}")
                self.enabled = False

    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        Transcribe un archivo de audio a texto.
        """
        if not self.enabled:
            return "Error: Whisper no está disponible."

        self._init_model()
        
        if not self._model:
            return "Error: No se pudo inicializar el modelo de voz."

        try:
            segments, info = self._model.transcribe(audio_path, beam_size=5)
            text = " ".join([segment.text for segment in segments]).strip()
            logger.info(f"Transcripción completada ({info.language}): {text}")
            return text
        except Exception as e:
            logger.error(f"Error en transcripción: {e}")
            return f"Error en transcripción: {str(e)}"
