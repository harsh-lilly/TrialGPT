import requests

# Base URL for the API
BASE_URL = "https://clinicaltrials.gov/api/v2"

# Example: Fetch version information
def fetch_version():
    url = f"{BASE_URL}/version"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Parse JSON response
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Example: Fetch trials
def fetch_trials():
    url = f"{BASE_URL}/studies/NCT00841061"
    # params = {
    #     "condition": "diabetes",  # Example search condition
    #     "status": "recruiting",  # Example search status
    #     "limit": 5,             # Limit results
    # }
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    print("Fetching API Version...")
    version_info = fetch_version()
    print(version_info)

    print("\nFetching Trials...")
    trial_data = fetch_trials()
    for key, value in trial_data.items():
        print(f'{key}: {value}')
        # print(value)
    # print(trial_data)