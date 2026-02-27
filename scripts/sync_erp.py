import sys
from pathlib import Path
import os

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from web_app.services import load_configuration, get_product_manager

def run_sync():
    print("Starting ERP Synchronization...")
    
    # Path to ERP data
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    csv_path = PROJECT_ROOT / "data" / "erp_inventory.csv"
    
    if not csv_path.exists():
        print(f"Error: ERP file not found at {csv_path}")
        return

    pm = get_product_manager()
    success = pm.sync_with_erp(str(csv_path))
    
    if success:
        print("ERP Synchronization COMPLETED successfully.")
    else:
        print("ERP Synchronization FAILED.")

if __name__ == "__main__":
    run_sync()
