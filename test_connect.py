import requests
import sys

url = "http://127.0.0.1:1234/v1/models"
print(f"Testing connection to {url}...")

try:
    resp = requests.get(url, timeout=5)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text[:200]}")
    if resp.status_code == 200:
        print("SUCCESS: Connection established.")
    else:
        print("FAILURE: Connection made but returned error.")
except Exception as e:
    print(f"CRITICAL FAILURE: {e}")
