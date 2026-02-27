import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from web_app.services import get_orchestrator, get_tool_manager, get_product_manager
from modules.db_manager import DBManager

def test_cart_flow():
    print("Testing Cart Flow Logic...")
    
    orchestrator = get_orchestrator()
    tm = get_tool_manager()
    pm = get_product_manager()
    
    # 1. Test direct tool execution
    print("\n--- Step 1: Direct Tool Execution ---")
    sku = "SOFA-CHE-001"
    result = tm.execute_tool("add_to_cart", {"sku": sku, "quantity": 1})
    print(f"Tool Output: {result}")
    
    if "[CART_ACTION]" in result and sku in result:
        print("✅ SUCCESS: Tool returned correct tag for UI interception.")
    else:
        print("❌ FAILURE: Tool did not return expected tag.")

    # 2. Test AI Routing
    print("\n--- Step 2: AI Routing Analysis ---")
    queries = [
        "Añade el sofá chesterfield al carrito",
        "Pon una mesa de roble en mi cesta",
        "Quiero comprar 2 sillas eames"
    ]
    
    for q in queries:
        print(f"Query: '{q}'")
        res = orchestrator.route_request(q, {"role": "ADMIN"})
        print(f"Target: {res['tool']}, ToolName: {res.get('tool_name')}")
        
        if res['tool'] == 'TOOL_CALL' and res.get('tool_name') == 'add_to_cart':
             print("✅ SUCCESS: Correctly routed to add_to_cart.")
        elif res['tool'] == 'PRODUCT_ADVISOR':
             print("⚠️ NOTE: Routed to advisor (might happen if intent is ambiguous).")
        else:
             print("❌ FAILURE: Incorrect routing.")

    # 3. Test inventory check integration
    print("\n--- Step 3: Inventory Check ---")
    inv_result = tm.execute_tool("check_inventory", {"sku": sku})
    print(f"Inventory Result: {inv_result}")
    if "Stock" in inv_result and "Precio" in inv_result:
        print("✅ SUCCESS: Inventory check works.")

if __name__ == "__main__":
    test_cart_flow()
