import os

files = [
    r'c:\Users\Usuario\Desktop\Repositorio Anas\AutoOCR_FinalVersion\web_app\app.py',
    r'c:\Users\Usuario\Desktop\Repositorio Anas\AutoOCR_FinalVersion\modules\db_manager.py',
    r'c:\Users\Usuario\Desktop\Repositorio Anas\AutoOCR_FinalVersion\modules\rag_manager.py'
]

targets = [
    ('\\"?\\"', '"?"'),
    ('\\" + db.placeholder + \\"', '" + db.placeholder + "'),
    ('\\"ORDER BY', '"ORDER BY'),
    ('OFFSET \\"', 'OFFSET "'),
    ('replace(\\"?\\"', 'replace("?"'),
    ('\\"SELECT', '"SELECT'),
    ('id_doc = \\"', 'id_doc = "'),
    ('\\"UPDATE', '"UPDATE'),
    ('\\"DELETE', '"DELETE'),
    ('\\"SET', '"SET'),
    ('WHERE id = \\"', 'WHERE id = "'),
    ('\\"\\"', '""'),
    ('\\"\\"\\"', '"""'),
    ('\\"', '"'), # Generic fallback for backslash and quote
]

for file_path in files:
    if not os.path.exists(file_path): continue
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    # First special case for the problematic line in db_manager
    new_content = new_content.replace('replace(\\"?\\", self.placeholder)', 'replace("?", self.placeholder)')
    
    # Generic cleaning of backslash-quote combos that are likely errors
    new_content = new_content.replace('\\"', '"')
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Deep cleaned {file_path}")

print("Deep cleaning finished.")
