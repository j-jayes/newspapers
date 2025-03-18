import requests

# Define the Search API URL
SEARCH_API_URL = "https://data.kb.se/search/"

# Updated Query Parameters (now including `service_selection`)
params = {
    "q": "Norrländska korrespondenten",
    "from": "1871-01-01",
    "to": "1871-12-10",
    "limit": 5,  # Get more results
    "_sort": "datePublished",
    "service_selection": "data.kb.se"  # Adding this parameter
}

# Headers
headers = {
    "Accept": "application/json"
}

# Step 1: Search for the newspaper issue
response = requests.get(SEARCH_API_URL, params=params, headers=headers)

# Try parsing as JSON
try:
    data = response.json()
    print(f"Status Code: {response.status_code}")
    print(f"Total Results: {data.get('total', 0)}")
    if data.get("hits"):
        print("First 3 Results:", data["hits"][:3])  # Print first 3 results
    else:
        print("No results found for Dagens Nyheter on 1900-01-01.")
except requests.exceptions.JSONDecodeError:
    print("Error: Response is not valid JSON")


# save data to temp.json
import json
with open("temp.json", "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)