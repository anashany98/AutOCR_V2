import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from web_app.services import load_configuration, get_orchestrator, get_db

def test_advisor():
    print("Initializing services...")
    config = load_configuration()
    
    orchestrator = get_orchestrator()
    if not orchestrator:
        print("Failed to initialize Orchestrator.")
        return

    # Mock user context
    context = {"role": "CLIENTE"}

    queries = [
        "Quiero comprar un sofá azul",
        "Busco una mesa de comedor grande",
        "Necesito sillas baratas"
    ]

    for q in queries:
        print(f"\n--- Testing Query: '{q}' ---")
        route = orchestrator.route_request(q, context)
        print(f"Route: {route['tool']}")
        
        if route['tool'] == 'PRODUCT_ADVISOR':
            res = orchestrator.handle_product_advice(q, context)
            print(f"Answer: {res.get('answer')}")
            print(f"Found {len(res.get('results', []))} products.")
            for p in res.get('results', []):
                print(f" - {p['name']} ({p['price']}€)")
        else:
            print("Did not route to PRODUCT_ADVISOR")

if __name__ == "__main__":
    test_advisor()
