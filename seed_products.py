import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.db_manager import DBManager
from modules.product_manager import ProductManager
from web_app.services import get_db

def seed():
    # Initialize DB and Product Manager
    # We must ensure we use the same DB instance config
    # Since this is a standalone script, we need to initialize properly
    
    # Just creating a new DBManager pointing to the data folder
    db = DBManager(config={"database": {"type": "sqlite", "path": "data/digitalizerai.db"}})
    # Or rely on get_db if flask app context is not needed, but get_db might need app context.
    # Safe bet is manual init or using the exact path we know.
    
    pm = ProductManager(db)
    
    products = [
        {
            "sku": "FUR-001",
            "name": "Sillón Orejero 'Velvet Royal'",
            "description": "Sillón clásico renovado con tapicería de terciopelo azul noche de alta resistencia (30.000 ciclos Martindale). Estructura de madera de haya maciza y patas acabadas en nogal oscuro. Incluye cojín lumbar. Dimensiones: 110x85x90cm. Ideal para zonas de lectura o salones elegantes.",
            "price": 450.00,
            "stock": 12,
            "image_url": "/static/products/velvet_royal.jpg"
        },
        {
            "sku": "FUR-002",
            "name": "Sofá Modular 'Cloud Nine' 3 Plazas",
            "description": "Sofá de diseño modular contemporáneo en tela bouclé color crema. Relleno de espuma de alta densidad + capa de plumas para máximo confort. Fundas desenfundables y lavables. Sistema de unión invisible entre módulos. Perfecto para espacios modernos y minimalistas.",
            "price": 1299.00,
            "stock": 5,
            "image_url": "/static/products/cloud_nine.jpg"
        },
        {
            "sku": "CUR-001",
            "name": "Cortinas Opacas 'Blackout Total' - Gris Perla",
            "description": "Set de 2 paneles de cortinas térmicas y opacas. Bloquean el 100% de la luz solar y reducen el ruido exterior. Tejido de triple capa con textura de lino sintético. Ojales metálicos inoxidables de 4cm. Medidas por panel: 140x260cm. Ahorro energético garantizado.",
            "price": 89.90,
            "stock": 50,
            "image_url": "/static/products/blackout_grey.jpg"
        },
        {
            "sku": "CUR-002",
            "name": "Visillos 'Brisa Marina' - Lino Natural",
            "description": "Visillos translúcidos confeccionados en 100% lino lavado europeo. Permiten el paso de luz tamizada creando un ambiente cálido y natural. Caída elegante y vaporosa. Color arena suave. Sistema de colgado dual (barra o riel). Medidas: 200x280cm.",
            "price": 120.00,
            "stock": 25,
            "image_url": "/static/products/linen_breeze.jpg"
        },
        {
            "sku": "FUR-003",
            "name": "Mesa de Comedor Extensible 'Nordic Oak'",
            "description": "Mesa de comedor de roble macizo con acabado al aceite mate. Diseño escandinavo de líneas puras. Sistema de extensión mariposa central oculto, pasa de 160cm a 220cm fácilmente. Capacidad para 6-10 comensales. Resistente a manchas y calor moderado.",
            "price": 850.00,
            "stock": 8,
            "image_url": "/static/products/nordic_table.jpg"
        },
         {
            "sku": "FUR-004",
            "name": "Silla de Comedor 'Eames Replica' - Pack 4",
            "description": "Pack de 4 sillas estilo nórdico. Asiento de polipropileno ergonómico en color blanco mate. Patas de madera de haya con refuerzos metálicos negros tipo Eiffel. Soporta hasta 120kg. Fáciles de limpiar y montar. Un clásico del diseño para tu comedor.",
            "price": 149.99,
            "stock": 100,
            "image_url": "/static/products/eames_pack.jpg"
        },
        {
            "sku": "CUR-003",
            "name": "Estor Enrollable Motorizado 'SmartBlind'",
            "description": "Estor enrollable de tejido screen (visibilidad 5%). Motor silencioso recargable por USB-C. Compatible con Alexa y Google Home. Color blanco técnico que repele el polvo. Medidas personalizables (stock estándar 120x200cm). Incluye mando a distancia.",
            "price": 199.50,
            "stock": 15,
            "image_url": "/static/products/smart_blind.jpg"
        }
    ]
    
    print(f"Seeding {len(products)} products...")
    for p in products:
        pm.add_product(
            p["sku"],
            p["name"],
            p["description"],
            p["price"],
            p["stock"],
            p["image_url"]
        )
    print("Done!")

if __name__ == "__main__":
    seed()
