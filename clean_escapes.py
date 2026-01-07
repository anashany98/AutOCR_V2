import os

file_path = r'c:\Users\Usuario\Desktop\Repositorio Anas\AutoOCR_FinalVersion\web_app\app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the incorrect escapes
content = content.replace('\\"?\\"', '"?"')
content = content.replace('\\" + db.placeholder + \\"', '" + db.placeholder + "')
content = content.replace('\\"ORDER BY', '"ORDER BY')
content = content.replace('OFFSET \\"', 'OFFSET "')
content = content.replace('replace(\\"?\\"', 'replace("?"')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Cleaned up escapes in app.py")
