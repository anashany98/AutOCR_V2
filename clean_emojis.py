import os

files_to_clean = [
    r'c:\Users\Usuario\Desktop\Repositorio Anas\AutoOCR_FinalVersion\web_app\app.py',
    r'c:\Users\Usuario\Desktop\Repositorio Anas\AutoOCR_FinalVersion\modules\ocr_manager.py',
    r'c:\Users\Usuario\Desktop\Repositorio Anas\AutoOCR_FinalVersion\modules\paddle_singleton.py'
]

emojis = ['🧩', '🧠', '⬇️', '🚀', '🔥', '✅', '❌', '⚠️', '⚙️']

for file_path in files_to_clean:
    if not os.path.exists(file_path):
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for emoji in emojis:
        new_content = new_content.replace(emoji, '')
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Cleaned emojis from {file_path}")
    else:
        print(f"No emojis found in {file_path}")
