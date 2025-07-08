import json

new_file = 'storage/trials_data.json'
original_file = '../TrialGPT-Demo-V2/trial_info.json'

with open(new_file, 'r') as file:
    new = json.load(file)

with open(original_file, 'r') as file:
    original = json.load(file)

newKeys = set()

for study in new.get("studies", []):
    key = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId", {})
    newKeys.add(key)


orgKeys = set(original.keys())



commonKeys = set()

for key in newKeys:
    if key in orgKeys:
        commonKeys.add(key)

print(commonKeys)


print(f"Total number of trials in Original Dataset: {len(orgKeys)}")
print(f"Total number of trials in New Dataset: {len(newKeys)}")
print(f"Total common keys between two datasets are: {len(commonKeys)}")

