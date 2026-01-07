import requests
import json

try:
    response = requests.post(
        "http://localhost:8000/api/chat",
        json={"query": "test query"},
        timeout=10
    )
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print("DEBUG INFO FROM SERVER:")
    print(json.dumps(data.get("debug"), indent=2))
    print(f"Number of results: {len(data.get('results', []))}")
    if data.get("answer"):
        print(f"Answer: {data.get('answer')}")
except Exception as e:
    print(f"Error: {e}")
