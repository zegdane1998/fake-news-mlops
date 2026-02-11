import requests

# Test with a real political headline
test_data = {"text": "New policy announced for Istanbul urban transformation projects."}

try:
    response = requests.post("http://127.0.0.1:8000/predict", json=test_data)
    print("Response from API:", response.json())
except Exception as e:
    print("Connection failed. Is the server running? Error:", e)