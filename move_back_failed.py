import os
import shutil
from pathlib import Path

def move_back():
    project_root = Path(__file__).resolve().parent
    errors_dir = project_root / "errors"
    input_dir = project_root / "input"
    
    if not input_dir.exists():
        os.makedirs(input_dir, exist_ok=True)
        
    count = 0
    # Search for all PDFs and images in errors dir
    for file in errors_dir.glob("**/*"):
        if file.is_file() and file.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".xlsx"]:
            # Skip if it's in CRASH_QUARANTINE
            if "CRASH_QUARANTINE" in str(file):
                continue
            
            dest = input_dir / file.name
            print(f"Moving {file.name} to {input_dir}")
            try:
                shutil.move(str(file), str(dest))
                count += 1
            except Exception as e:
                print(f"Failed to move {file.name}: {e}")
                
    print(f"Moved {count} files back to input.")

if __name__ == "__main__":
    move_back()
