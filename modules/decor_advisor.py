from typing import List, Dict

class DecorAdvisor:
    """
    Simple advice engine based on detected visual tags.
    """
    
    def __init__(self):
        # Rules: If Key Tag is present, suggest Value.
        self.rules = {
            "Madera Oscura": "Combina con tonos crema o textiles claros para dar luminosidad.",
            "Madera Roble (Oak)": "Va perfecto con verdes naturales y tonos tierra.",
            "Estilo Industrial": "Añade calidez con plantas y textiles suaves.",
            "Estilo Nórdico": "Mantén la paleta neutra y añade texturas naturales.",
            "Tela Terciopelo": "Aporta sofisticación. Evita recargar con demasiados estampados.",
            "Color Azul Marino": "Contrasta genial con mostaza o dorado.",
            "Color Verde Esmeralda": "Combina con maderas oscuras y latón.",
            "Color Terracota": "Ideal para ambientes rústicos o bohemios. Usa madera natural.",
            "Estilo Minimalista": "Menos es más. Añade una sola pieza de arte grande."
        }

    def generate_advice(self, tags: List[str]) -> List[str]:
        advice = []
        # Pre-process tags to remove probability scores "Tag (90%)" -> "Tag"
        clean_tags = [t.split('(')[0].strip() for t in tags]
        
        for tag in clean_tags:
            if tag in self.rules:
                advice.append(f"💡 {tag}: {self.rules[tag]}")
                
        # Fallback combinations
        has_color = any("Color" in t for t in clean_tags)
        has_wood = any("Madera" in t for t in clean_tags)
        
        if has_color and not has_wood:
            advice.append("🎨 Tip: Introduce elementos de madera para dar calidez al color.")
        
        return list(set(advice)) # Deduplicate

    def generate_ai_advice(self, caption: str, objects: List[str], llm_client) -> str:
        """
        Uses LLM to provide professional decor advice based on visual analysis.
        """
        if not llm_client:
            return "El asesor inteligente está desactivado por el momento."

        prompt = (
            f"Como experto en diseño de interiores y arquitectura, analiza estos datos visuales y da 3 consejos cortos y profesionales.\n"
            f"ESCENA: {caption}\n"
            f"OBJETOS DETECTADOS: {', '.join(objects)}\n"
            f"Responde en español de forma inspiracional y técnica."
        )

        try:
            # We use analyze_document generic method as a base for custom prompts if supported
            # or add a specific chat-like method. Let's use custom prompt logic.
            result = llm_client.analyze_document(text=prompt, doc_type="Diseño de Interiores", reason="Asesoría Estética")
            # Usually analyze_document returns JSON, we might want just text for advice
            if isinstance(result, dict) and "analysis" in result:
                return result["analysis"]
            return str(result)
        except Exception as e:
            return f"Error al generar asesoría: {e}"
