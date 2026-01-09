import logging
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ValidationError

# --- Contratos de Datos (Pydantic) ---

class MetricasOCR(BaseModel):
    confianza_media: float = Field(..., ge=0.0, le=1.0)
    bloques_baja_confianza: int = Field(..., ge=0)
    texto_legible_global: bool

class IndicadoresGraficos(BaseModel):
    escritura_mano_detectada: bool
    dibujos_o_lineas_no_textuales: bool
    estructura_visual_irregular: bool

class InterpretationRequest(BaseModel):
    document_id: str
    tipo_archivo: str  # "pdf" | "imagen"
    paginas: int
    es_pdf_nativo: bool
    clasificacion_previa: str
    metricas_ocr: MetricasOCR
    indicadores_graficos: IndicadoresGraficos
    resumen_ocr: Optional[str] = ""

class InterpretationResponse(BaseModel):
    activar_interpretacion_avanzada: bool
    accion: Literal["continuar_pipeline_estandar", "invocar_modulo_interpretacion"]
    motivo: str
    tipo_interpretacion: Optional[str] = None
    datos_a_enviar: Optional[Dict[str, Any]] = None
    confianza_decision: float = Field(..., ge=0.0, le=1.0)


# --- Cliente LLM Desacoplado (Interfaz) ---

class AbstractLLMClient:
    """
    Interfaz abstracta para el cliente LLM.
    En el futuro, aquí se inyectaría la implementación real (OpenAI, Anthropic, Local, etc).
    """
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Este cliente es abstracto.")


# --- Router Principal ---

class AdvancedInterpretationRouter:
    """
    Módulo aislado que decide si activar interpretación avanzada (LLM).
    NO realiza OCR. NO modifica datos. Solo decide.
    """
    def __init__(self, llm_client: Optional[AbstractLLMClient] = None, logger: Optional[logging.Logger] = None):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuración de umbrales
        self.UMBRAL_CONFIANZA_BAJA = 0.70  # Si es menor, sospechamos
        self.UMBRAL_CONFIANZA_CRITICA = 0.40 # Si es menor, casi seguro basura
        
        # Tipos que casi siempre requieren IA
        self.TIPOS_COMPLEJOS = {"plano", "boceto", "diagrama", "manuscrito"}

    def evaluate_document(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Punto de entrada único. Evalúa y devuelve decisión JSON estricta.
        Nunca lanza excepciones hacia afuera (atrapa todo y devuelve fallback).
        """
        try:
            # 1. Validación Estricta
            try:
                request = InterpretationRequest(**input_data)
            except ValidationError as ve:
                self.logger.warning(f"Error de validación en entrada del router: {ve}")
                return self._fallback_response(f"Error de validación: {ve}")

            # 2. Regla de Precedencia Absoluta: PDF Nativo con texto legible
            if request.es_pdf_nativo and request.metricas_ocr.confianza_media > 0.9 and request.metricas_ocr.texto_legible_global:
                 return self._build_response(
                    activar=False,
                    motivo="Documento digital nativo con texto legible y alta confianza. OCR clásico suficiente.",
                    confianza=0.99
                )

            # 3. Evaluación de Indicadores
            
            # A. Escritura a mano
            if request.indicadores_graficos.escritura_mano_detectada:
                 return self._build_response(
                    activar=True,
                    motivo="Escritura manual detectada. Requiere interpretación semántica.",
                    confianza=0.95,
                    tipo="manuscrito"
                )

            # B. Tipos complejos conocidos
            if request.clasificacion_previa.lower() in self.TIPOS_COMPLEJOS:
                return self._build_response(
                    activar=True,
                    motivo=f"Clasificación '{request.clasificacion_previa}' requiere análisis visual avanzado.",
                    confianza=0.90,
                    tipo="visual_complex"
                )

            # C. Baja Confianza del OCR
            confianza = request.metricas_ocr.confianza_media
            if confianza < self.UMBRAL_CONFIANZA_CRITICA:
                 return self._build_response(
                    activar=True,
                    motivo=f"Confianza OCR crítica ({confianza:.2f}). Posible imagen sin texto o ruido, o escritura ilegible.",
                    confianza=0.85,
                    tipo="recuperacion_ruido"
                )
            
            if confianza < self.UMBRAL_CONFIANZA_BAJA and request.indicadores_graficos.dibujos_o_lineas_no_textuales:
                 return self._build_response(
                    activar=True,
                    motivo="Confianza media-baja con elementos gráficos. Posible documento mixto.",
                    confianza=0.75,
                    tipo="mixto"
                )

            # 4. Por defecto: Continuar Pipeline Estándar
            return self._build_response(
                activar=False,
                motivo="OCR clásico produjo resultados aceptables dentro de los umbrales.",
                confianza=0.90
            )

        except Exception as e:
            self.logger.error(f"Error inesperado en AdvancedInterpretationRouter: {e}", exc_info=True)
            return self._fallback_response(f"Error interno: {str(e)}")

    def _build_response(self, activar: bool, motivo: str, confianza: float, tipo: str = None) -> Dict[str, Any]:
        resp = InterpretationResponse(
            activar_interpretacion_avanzada=activar,
            accion="invocar_modulo_interpretacion" if activar else "continuar_pipeline_estandar",
            motivo=motivo,
            tipo_interpretacion=tipo,
            datos_a_enviar=None, # Opcional, se llenaría si tuviéramos datos extra
            confianza_decision=confianza
        )
        return resp.model_dump()

    def _fallback_response(self, razon: str) -> Dict[str, Any]:
        """Respuesta segura en caso de pánico."""
        return {
            "activar_interpretacion_avanzada": False,
            "accion": "continuar_pipeline_estandar",
            "motivo": f"Fallback por error: {razon}",
            "tipo_interpretacion": None,
            "datos_a_enviar": None,
            "confianza_decision": 0.0
        }
