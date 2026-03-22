import requests

# Test with a real political headline
test_headline = "New policy announced for Istanbul urban transformation projects."

try:
    response = requests.post("http://127.0.0.1:8000/analyze", data={"headline": test_headline})
    print("Status:", response.status_code)
    print("Response from API:", response.text[:200])
except Exception as e:
    print("Connection failed. Is the server running? Error:", e)