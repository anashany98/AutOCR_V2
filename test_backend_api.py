from web_app.app import app
import sys

print("Testing /api/status/llm endpoint...")
try:
    with app.test_client() as client:
        resp = client.get("/api/status/llm")
        print(f"Status Code: {resp.status_code}")
        print(f"Data: {resp.get_data(as_text=True)}")
except Exception as e:
    print(f"CRITICAL BACKEND FAILURE: {e}")
    import traceback
    traceback.print_exc()
