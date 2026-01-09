import os

EXTS = {'.py', '.html', '.css', '.js', '.yaml', '.bat'}
SKIP = {'venv', '.git', '.gemini', '__pycache__', 'data', 'node_modules', 'rag_index', 'logs', 'db', 'venv311', 'tools'}

total_lines = 0
file_count = 0
by_ext = {}

root = r"c:\Users\Usuario\Desktop\Repositorio Anas\AutoOCR_FinalVersion"

for dirpath, dirnames, filenames in os.walk(root):
    # Filter directories in place
    dirnames[:] = [d for d in dirnames if d not in SKIP and not d.startswith('.')]
    
    for f in filenames:
        ext = os.path.splitext(f)[1].lower()
        if ext in EXTS or f == 'Dockerfile':
            full_path = os.path.join(dirpath, f)
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as fh:
                    lines = len(fh.readlines())
                    total_lines += lines
                    file_count += 1
                    by_ext[ext] = by_ext.get(ext, 0) + lines
            except Exception:
                pass

print(f"Total Files: {file_count}")
print(f"Total Lines: {total_lines}")
print("Breakdown:")
for k, v in by_ext.items():
    print(f"  {k}: {v}")
