import requests
import json

# Base URL for the API
BASE_URL = "https://clinicaltrials.gov/api/v2"
OUTPUT_FILE = "storage/trials_data.json"

def fetch_trials():
    """Fetches all trial data using pageToken-based pagination."""
    trials = []
    page_token = None  # Start without a token

    while True:
        url = f"{BASE_URL}/studies?filter.advanced=Eli+Lilly+and+Company"
        if page_token:
            url += f"&pageToken={page_token}"

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            studies = data.get("studies", [])

            if studies:
                trials.extend(studies)

            # Get the next page token
            page_token = data.get("nextPageToken")

            # Stop if there are no more pages
            if not page_token:
                break
        else:
            print(f"Error fetching trials: {response.status_code}")
            print(response.text)
            break

    return {"studies": trials}  # Wrap in a dictionary

def save_to_file(data, filename):
    """Saves the data to a JSON file."""
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    print("Fetching Paginated Trials...")
    trial_data = fetch_trials()
    
    if trial_data:
        save_to_file(trial_data, OUTPUT_FILE)
