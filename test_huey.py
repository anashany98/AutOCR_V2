from modules.tasks import huey, process_document_task
import time

print(f"Pending before: {len(huey.pending())}")

# Enqueue a dummy task (we won't actually process a file to avoid errors, just checking queue)
# But process_document_task expects arguments.
# We can just check pending.

# Let's inspect the DB file size
import os
db_path = 'data/huey_db.db'
if os.path.exists(db_path):
    print(f"DB Size: {os.path.getsize(db_path)} bytes")
else:
    print("DB does not exist!")

print("Tasks in queue:")
try:
    for task in huey.pending():
        print(f" - {task}")
except Exception as e:
    print(f"Error reading pending: {e}")
