"""
Agente de Extracción Multi-Paso Inteligente.

Este módulo implementa un agente AI que:
1. Analiza el documento para determinar su tipo
2. Decide qué campos extraer basándose en el contexto
3. Utiliza el LLM para razonar sobre la estructura del documento
4. Valida y normaliza los resultados

Diferencia con smart_extractor.py:
- smart_extractor: extracción basada en reglas (regex) + fallback LLM
- extraction_agent: el AGENTE decide QUÉ extraer y CÓMO hacerlo
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Tipos de documento soportados."""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    ORDER = "order"
    REPORT = "report"
    TECHNICAL_PLAN = "technical_plan"
    UNKNOWN = "unknown"


@dataclass
class ExtractionPlan:
    """Plan de extracción generado por el agente."""
    document_type: DocumentType
    confidence: float
    fields_to_extract: List[str]
    reasoning: str
    extraction_strategy: str  # "regex", "llm", "hybrid"


@dataclass
class ExtractionResult:
    """Resultado de la extracción."""
    success: bool
    document_type: Optional[DocumentType] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    error: Optional[str] = None


class ExtractionAgent:
    """
    Agente inteligente de extracción de campos.
    
    Decide dinámicamente qué extraer basándose en el contenido
    del documento usando razonamiento LLM.
    """
    
    # Definiciones de esquemas por tipo de documento
    SCHEMA_DEFINITIONS = {
        DocumentType.INVOICE: {
            "required": ["date", "total", "vendor", "invoice_number"],
            "optional": ["due_date", "tax_id", "iva", "subtotal", "payment_terms"],
            "description": "Factura comercial con IVA, proveedor y número de factura"
        },
        DocumentType.RECEIPT: {
            "required": ["date", "total", "vendor"],
            "optional": ["items", "payment_method", "tip"],
            "description": "Ticket de compra minorista"
        },
        DocumentType.CONTRACT: {
            "required": ["parties", "start_date", "end_date"],
            "optional": ["penalities", "renewal_terms", "jurisdiction"],
            "description": "Contrato legal o comercial"
        },
        DocumentType.ORDER: {
            "required": ["order_number", "date", "vendor"],
            "optional": ["delivery_date", "items", "total"],
            "description": "Orden de compra o pedido"
        },
        DocumentType.REPORT: {
            "required": ["title", "date"],
            "optional": ["author", "summary", "sections"],
            "description": "Informe técnico o reporte"
        },
        DocumentType.TECHNICAL_PLAN: {
            "required": ["title", "scale"],
            "optional": ["author", "version", "date"],
            "description": "Plano técnico o dibujo técnico"
        },
    }
    
    def __init__(
        self,
        llm_client=None,
        config: Optional[Dict[str, Any]] = None,
        extractors: Optional[Dict[str, Callable]] = None
    ):
        """
        Inicializar el agente.
        
        Args:
            llm_client: Cliente LLM para razonamiento
            config: Configuración opcional
            extractors: Diccionario de funciones extractor por tipo de campo
        """
        self.config = config or {}
        self.llm_client = llm_client
        self._llm_checked = False
        
        # Extractores base (regex-based)
        self.extractors = extractors or {}
        
    def _get_llm_client(self):
        """Obtener cliente LLM lazily."""
        if self._llm_checked:
            return self.llm_client
        self._llm_checked = True
        
        if self.llm_client:
            return self.llm_client
            
        try:
            from web_app.services import get_llm_client
            self.llm_client = get_llm_client()
        except Exception as e:
            logger.warning(f"Could not get LLM client: {e}")
            
        return self.llm_client
    
    def analyze(self, text: str, blocks: Optional[List[Dict]] = None) -> ExtractionPlan:
        """
        Análisis inicial: determinar tipo de documento y plan de extracción.
        
        Args:
            text: Texto OCR del documento
            blocks: Bloques estructurados (opcional)
            
        Returns:
            ExtractionPlan con la estrategia a seguir
        """
        llm = self._get_llm_client()
        
        # Prompt de análisis
        analysis_prompt = self._build_analysis_prompt(text, blocks)
        
        if llm and getattr(llm, "enabled", False):
            try:
                # Usar LLM para análisis inteligente
                result = llm.analyze_document(
                    text=text[:3000],  # Limitar contexto
                    reason="determinar_tipo_documento",
                    doc_type="analysis"
                )
                
                if result.get("success"):
                    plan = self._parse_llm_analysis(result.get("analysis", {}))
                    if plan:
                        logger.info(f"Agent chose document type: {plan.document_type}")
                        return plan
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
        
        # Fallback: análisis basado en reglas
        return self._rule_based_analysis(text)
    
    def _build_analysis_prompt(self, text: str, blocks: Optional[List[Dict]]) -> str:
        """Construir prompt para análisis LLM."""
        # Primeros 1500 caracteres como muestra
        text_sample = text[:1500].replace("```", "").strip()
        
        return f"""Analiza este documento y determina:
1. Tipo de documento (invoice, receipt, contract, order, report, technical_plan)
2. Qué campos son relevantes extraer
3. Estrategia de extracción recomendada

Documento:
{text_sample}

Responde en JSON:
{{
    "document_type": "...",
    "confidence": 0.0-1.0,
    "fields_to_extract": ["field1", "field2", ...],
    "reasoning": "explicación breve",
    "extraction_strategy": "regex|llm|hybrid"
}}"""
    
    def _parse_llm_analysis(self, analysis: Any) -> Optional[ExtractionPlan]:
        """Parsear respuesta del LLM a ExtractionPlan."""
        try:
            if isinstance(analysis, str):
                data = json.loads(analysis)
            elif isinstance(analysis, dict):
                data = analysis
            else:
                return None
                
            doc_type = DocumentType(data.get("document_type", "unknown").lower())
            if doc_type not in DocumentType:
                doc_type = DocumentType.UNKNOWN
                
            return ExtractionPlan(
                document_type=doc_type,
                confidence=float(data.get("confidence", 0.5)),
                fields_to_extract=data.get("fields_to_extract", []),
                reasoning=data.get("reasoning", ""),
                extraction_strategy=data.get("extraction_strategy", "hybrid")
            )
        except Exception as e:
            logger.warning(f"Failed to parse LLM analysis: {e}")
            return None
    
    def _rule_based_analysis(self, text: str) -> ExtractionPlan:
        """Análisis basado en reglas cuando LLM no está disponible."""
        text_lower = text.lower()
        
        # Detectar tipo por keywords
        if any(k in text_lower for k in ["factura", "invoice", "facture"]):
            doc_type = DocumentType.INVOICE
        elif any(k in text_lower for k in ["ticket", "receipt", "recibo"]):
            doc_type = DocumentType.RECEIPT
        elif any(k in text_lower for k in ["contrato", "contract"]):
            doc_type = DocumentType.CONTRACT
        elif any(k in text_lower for k in ["pedido", "order", "orden de compra"]):
            doc_type = DocumentType.ORDER
        elif any(k in text_lower for k in ["informe", "report"]):
            doc_type = DocumentType.REPORT
        elif any(k in text_lower for k in ["plano", "planos", "drawing", "blueprint"]):
            doc_type = DocumentType.TECHNICAL_PLAN
        else:
            doc_type = DocumentType.UNKNOWN
            
        # Obtener schema
        schema = self.SCHEMA_DEFINITIONS.get(doc_type, {})
        
        return ExtractionPlan(
            document_type=doc_type,
            confidence=0.7 if doc_type != DocumentType.UNKNOWN else 0.3,
            fields_to_extract=schema.get("required", []) + schema.get("optional", [])[:5],
            reasoning=f"Rule-based detection: found {doc_type.value} keywords",
            extraction_strategy="hybrid"
        )
    
    def extract(
        self,
        text: str,
        plan: Optional[ExtractionPlan] = None,
        blocks: Optional[List[Dict]] = None
    ) -> ExtractionResult:
        """
        Ejecutar extracción basada en el plan.
        
        Args:
            text: Texto OCR
            plan: Plan de extracción (opcional, se genera si no se provee)
            blocks: Bloques estructurados
            
        Returns:
            ExtractionResult con campos extraídos
        """
        # Si no hay plan, analizar primero
        if plan is None:
            plan = self.analyze(text, blocks)
            
        if plan.document_type == DocumentType.UNKNOWN:
            return ExtractionResult(
                success=False,
                error="No se pudo determinar el tipo de documento"
            )
            
        # Obtener schema
        schema = self.SCHEMA_DEFINITIONS.get(plan.document_type, {})
        
        # Estrategia de extracción
        if plan.extraction_strategy == "llm":
            return self._extract_with_llm(text, plan, schema, blocks)
        elif plan.extraction_strategy == "regex":
            return self._extract_with_regex(text, plan, schema, blocks)
        else:  # hybrid
            return self._extract_hybrid(text, plan, schema, blocks)
    
    def _extract_with_llm(
        self,
        text: str,
        plan: ExtractionPlan,
        schema: Dict,
        blocks: Optional[List[Dict]]
    ) -> ExtractionResult:
        """Extracción usando LLM."""
        llm = self._get_llm_client()
        
        if not llm or not getattr(llm, "enabled", False):
            return self._extract_with_regex(text, plan, schema, blocks)
            
        # Obtener campos a extraer
        fields = plan.fields_to_extract or schema.get("required", [])
        
        try:
            # Usar smart_extract del LLM
            result = llm.smart_extract(text[:5000], fields)
            
            if result.get("success"):
                return ExtractionResult(
                    success=True,
                    document_type=plan.document_type,
                    fields=result.get("analysis", {}),
                    confidence=plan.confidence,
                    reasoning=f"LLM extraction: {plan.reasoning}"
                )
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            
        # Fallback
        return self._extract_with_regex(text, plan, schema, blocks)
    
    def _extract_with_regex(
        self,
        text: str,
        plan: ExtractionPlan,
        schema: Dict,
        blocks: Optional[List[Dict]]
    ) -> ExtractionResult:
        """Extracción usando regex (delegar a extractores existentes)."""
        from .smart_extractor import FieldExtractor
        
        extractor = FieldExtractor(self.config)
        extracted = extractor.extract_fields(text, blocks)
        
        return ExtractionResult(
            success=True,
            document_type=plan.document_type,
            fields=extracted,
            confidence=plan.confidence * 0.8,  # Lower confidence for regex-only
            reasoning=f"Regex extraction: {plan.reasoning}"
        )
    
    def _extract_hybrid(
        self,
        text: str,
        plan: ExtractionPlan,
        schema: Dict,
        blocks: Optional[List[Dict]]
    ) -> ExtractionResult:
        """Extracción híbrida: regex primero, LLM para completar."""
        # Paso 1: Extraer con regex
        regex_result = self._extract_with_regex(text, plan, schema, blocks)
        
        # Paso 2: Usar LLM para completar campos faltantes
        llm = self._get_llm_client()
        
        if llm and getattr(llm, "enabled", False):
            missing_fields = [
                f for f in schema.get("required", [])
                if f not in regex_result.fields or not regex_result.fields.get(f)
            ]
            
            if missing_fields:
                try:
                    # Intentar completar con LLM
                    llm_result = llm.smart_extract(text[:5000], missing_fields)
                    
                    if llm_result.get("success"):
                        llm_fields = llm_result.get("analysis", {})
                        
                        # Merge: regex tiene prioridad
                        for field, value in llm_fields.items():
                            if field not in regex_result.fields:
                                regex_result.fields[field] = {
                                    "value": value,
                                    "source": "llm"
                                }
                                
                        regex_result.reasoning += " + LLM fill"
                except Exception as e:
                    logger.debug(f"LLM fill failed: {e}")
        
        # Paso 3: Detectar anomalías
        anomalies = self._detect_anomalies(regex_result.fields, schema)
        regex_result.anomalies = anomalies
        
        return regex_result
    
    def _detect_anomalies(self, fields: Dict, schema: Dict) -> List[str]:
        """Detectar anomalías en campos extraídos."""
        anomalies = []
        
        # Verificar campos requeridos
        for required_field in schema.get("required", []):
            if required_field not in fields or not fields.get(required_field):
                anomalies.append(f"missing_required_{required_field}")
        
        # Verificar totales negativos o cero
        if "total" in fields:
            try:
                total_val = float(fields["total"].get("value", 0))
                if total_val <= 0:
                    anomalies.append("invalid_total_zero_or_negative")
            except (ValueError, TypeError):
                anomalies.append("invalid_total_format")
        
        # Verificar fechas futuras
        if "date" in fields:
            from datetime import datetime
            try:
                date_str = fields["date"].get("value", "")
                if date_str:
                    parsed = datetime.fromisoformat(date_str.replace("/", "-"))
                    if parsed.year > datetime.now().year + 1:
                        anomalies.append("suspicious_future_date")
            except Exception:
                pass
                
        return anomalies
    
    def process(self, text: str, blocks: Optional[List[Dict]] = None) -> ExtractionResult:
        """
        Método principal: análisis + extracción en un paso.
        
        Args:
            text: Texto OCR del documento
            blocks: Bloques estructurados (opcional)
            
        Returns:
            ExtractionResult completo
        """
        # Análisis
        plan = self.analyze(text, blocks)
        
        # Extracción
        return self.extract(text, plan, blocks)


# Alias para compatibilidad
AgenticExtractor = ExtractionAgent


__all__ = [
    "ExtractionAgent",
    "AgenticExtractor", 
    "ExtractionPlan",
    "ExtractionResult",
    "DocumentType",
]
