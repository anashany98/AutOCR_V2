import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.product_manager import ProductManager
from modules.db_manager import DBManager
from web_app.services import load_configuration

def seed():
    
    # Use service locator to ensure we test the actual app wiring
    from web_app.services import get_product_manager
    pm = get_product_manager()
    
    if pm.vision_manager:
        print("VisionManager is ACTIVE in ProductManager.")
    else:
        print("WARNING: VisionManager is NOT active in ProductManager.")
    
    print("Seeding products...")
    
    products = [
        {
            "sku": "SOFA-CHE-001",
            "name": "Sofá Chesterfield Azul",
            "description": "Sofá clásico estilo Chesterfield tapizado en terciopelo azul marino. Capitoneado profundo y patas de madera oscura.",
            "price": 1299.00,
            "stock": 5,
            "image_url": "https://example.com/sofa_blue.jpg",
            "attributes": {"color": "Azul Marino", "material": "Terciopelo", "style": "Chesterfield", "seats": 3},
            "category": "Sofa",
            "tags": ["clásico", "elegante", "salón"]
        },
        {
            "sku": "MESA-ROB-002",
            "name": "Mesa Comedor Roble Macizo",
            "description": "Mesa de comedor rectangular en madera de roble natural. Acabado mate. Extensible hasta 220cm.",
            "price": 850.00,
            "stock": 10,
            "image_url": "https://example.com/table_oak.jpg",
            "attributes": {"color": "Roble Natural", "material": "Madera Maciza", "style": "Nórdico", "seats": 6},
            "category": "Mesa",
            "tags": ["comedor", "nórdico", "madera"]
        },
        {
            "sku": "SILLA-EAM-003",
            "name": "Pack 4 Sillas Eames Replica",
            "description": "Conjunto de 4 sillas estilo Eames en color blanco. Patas de madera de haya y estructura metálica negra.",
            "price": 120.00,
            "stock": 50,
            "image_url": "https://example.com/chairs_white.jpg",
            "attributes": {"color": "Blanco", "material": "Polipropileno", "style": "Moderno", "pack": 4},
            "category": "Silla",
            "tags": ["pack", "económico", "diseño"]
        }
    ]
    
    for p in products:
        print(f"Adding {p['name']}...")
        pm.add_product(
            sku=p["sku"],
            name=p["name"],
            description=p["description"],
            price=p["price"],
            stock=p["stock"],
            image_url=p["image_url"],
            attributes=p["attributes"],
            category=p["category"],
            tags=p["tags"]
        )
        
    print("Seeding complete.")

if __name__ == "__main__":
    seed()
