import sys
import traceback
try:
    from web_app.app import app, init_app
    print("App imported successfully!")
    init_app()
    print("init_app() completed successfully!")
except Exception as e:
    traceback.print_exc()
    print(f"\nError: {e}")
