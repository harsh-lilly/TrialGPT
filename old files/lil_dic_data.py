import json

# Load the original JSON file
with open("trials_data.json", "r") as file:
    data = json.load(file)

# Extract relevant data
result = {}
count = 0
for study in data.get("studies", []):
    identification = study.get("protocolSection", {}).get("identificationModule", {})
    nct_id = identification.get("nctId")
    brief_title = identification.get("briefTitle")
    official_title = identification.get("officialTitle")

    # Extracting 'Lilly_alias' from secondaryIdInfos
    secondary_ids = identification.get("secondaryIdInfos", [])
    lilly_alias = [entry["id"] for entry in secondary_ids if "id" in entry]

    if nct_id:  # Ensure NCTId exists
        result[nct_id] = {
            "briefTitle": brief_title,
            "officialTitle": official_title,
            "lillyAlias": lilly_alias  # Storing the list of IDs
        }
        count += 1

# Save the extracted data into a new JSON file
with open("filtered_studies.json", "w") as file:
    json.dump(result, file, indent=4)

print(f"Filtered JSON file has been created: filtered_studies.json with a total of {count} records.")

