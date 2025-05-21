import json

# File names
retrieved_trials_file = "retrieved_trials.json"
trial_info_file = "trial_info.json"
detailed_trials_file = "detailed_trials.json"

# Step 1: Load relevant trial IDs from retrieved_trials.json
with open(retrieved_trials_file, "r") as f:
    retrieved_trials = json.load(f)

relevant_trial_ids = retrieved_trials["retrieved_trials"]  # Assuming a list of IDs

# Step 2: Load the trial metadata from trial_info.json
with open(trial_info_file, "r") as f:
    trial_info = json.load(f)

# Step 3: Fetch metadata for relevant trials
detailed_trials = [
    {"trial_id": trial_id, **trial_info[trial_id]}
    for trial_id in relevant_trial_ids
    if trial_id in trial_info
]

# Step 4: Save the detailed metadata to a new file
with open(detailed_trials_file, "w") as f:
    json.dump(detailed_trials, f, indent=4)

# Print a summary of the operation
print(f"Total trials in trial_info.json: {len(trial_info)}")
print(f"Relevant trial IDs in retrieved_trials.json: {len(relevant_trial_ids)}")
print(f"Matched detailed trials saved: {len(detailed_trials)}")
print(f"Detailed trial data saved to {detailed_trials_file}.")
