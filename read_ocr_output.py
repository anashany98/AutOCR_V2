import os

def read_output():
    path = "ocr_test_output.txt"
    if not os.path.exists(path):
        print("File not found.")
        return
    
    with open(path, "rb") as f:
        content = f.read()
    
    # Try different encodings common on Windows
    for enc in ["utf-8", "utf-16", "cp1252"]:
        try:
            text = content.decode(enc)
            print(f"--- CONTENT ({enc}) ---")
            print(text)
            return
        except:
            continue
    
    print("Could not decode file.")

if __name__ == "__main__":
    read_output()
