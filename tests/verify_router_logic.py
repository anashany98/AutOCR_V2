import sys
import os
import json
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.interpretation_manager import AdvancedInterpretationRouter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestRouter")

def run_test_case(name, input_data, expected_action):
    print(f"\n--- Running Test: {name} ---")
    router = AdvancedInterpretationRouter(logger=logger)
    result = router.evaluate_document(input_data)
    
    print(f"Input: {json.dumps(input_data, indent=2)}")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    if result["accion"] == expected_action:
        print("✅ PASS")
    else:
        print(f"❌ FAIL: Expected {expected_action}, got {result['accion']}")
        print(f"Reason: {result['motivo']}")

def main():
    # 1. Happy Path: Documento claro (Factura)
    data_invoice = {
        "document_id": "123",
        "tipo_archivo": "imagen",
        "paginas": 1,
        "es_pdf_nativo": False,
        "clasificacion_previa": "factura",
        "metricas_ocr": {
            "confianza_media": 0.95,
            "bloques_baja_confianza": 0,
            "texto_legible_global": True
        },
        "indicadores_graficos": {
            "escritura_mano_detectada": False,
            "dibujos_o_lineas_no_textuales": False,
            "estructura_visual_irregular": False
        },
        "resumen_ocr": "Factura 001. Total $100."
    }
    run_test_case("Clear Invoice", data_invoice, "continuar_pipeline_estandar")

    # 2. Casos de Activación: Manuscrito
    data_handwriting = {
        "document_id": "124",
        "tipo_archivo": "imagen",
        "paginas": 1,
        "es_pdf_nativo": False,
        "clasificacion_previa": "nota",
        "metricas_ocr": {
            "confianza_media": 0.60,
            "bloques_baja_confianza": 5,
            "texto_legible_global": False
        },
        "indicadores_graficos": {
            "escritura_mano_detectada": True, # TRIGGER
            "dibujos_o_lineas_no_textuales": False,
            "estructura_visual_irregular": True
        },
        "resumen_ocr": "..."
    }
    run_test_case("Handwritten Note", data_handwriting, "invocar_modulo_interpretacion")

    # 3. Baja Confianza
    data_low_conf = {
        "document_id": "125",
        "tipo_archivo": "imagen",
        "paginas": 1,
        "es_pdf_nativo": False,
        "clasificacion_previa": "otro",
        "metricas_ocr": {
            "confianza_media": 0.30, # TRIGGER
            "bloques_baja_confianza": 20,
            "texto_legible_global": False
        },
        "indicadores_graficos": {
            "escritura_mano_detectada": False,
            "dibujos_o_lineas_no_textuales": True,
            "estructura_visual_irregular": True
        }
    }
    run_test_case("Low Confidence Garbage", data_low_conf, "invocar_modulo_interpretacion")

    # 4. Regla Absoluta: PDF Nativo perfecto (aunque tenga un dibujo)
    data_native = {
        "document_id": "126",
        "tipo_archivo": "pdf",
        "paginas": 5,
        "es_pdf_nativo": True, # TRIGGER SKIP
        "clasificacion_previa": "informe",
        "metricas_ocr": {
            "confianza_media": 0.99,
            "bloques_baja_confianza": 0,
            "texto_legible_global": True
        },
        "indicadores_graficos": {
            "escritura_mano_detectada": False,
            "dibujos_o_lineas_no_textuales": True, # Ignored due to precedence
            "estructura_visual_irregular": False
        }
    }
    run_test_case("Native PDF Precedence", data_native, "continuar_pipeline_estandar")

    # 5. Error Handling (Missing fields)
    data_invalid = {
        "document_id": "127",
        # Missing required fields like metricas_ocr
    }
    run_test_case("Invalid Data Fallback", data_invalid, "continuar_pipeline_estandar")

if __name__ == "__main__":
    main()
