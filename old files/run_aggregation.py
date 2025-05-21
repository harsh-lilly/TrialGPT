__author__ = "qiao"

"""
TrialGPT: Aggregating and Ranking Clinical Trial Scores for a Single Patient.
"""

import json
from openai import AzureOpenAI
import os

# Azure OpenAI client setup
client = AzureOpenAI(
    api_version="2023-09-01-preview",
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Hard-coded patient note
with open('input.json', 'r') as file:
        data = json.load(file)

PATIENT_NOTE = data['patient_note']

def convert_criteria_pred_to_string(prediction, trial_info):
    """Convert prediction data into a readable string format."""
    output = ""
    for inc_exc in ["inclusion", "exclusion"]:
        idx2criterion = {}
        criteria = trial_info[inc_exc + "_criteria"].split("\n\n")

        idx = 0
        for criterion in criteria:
            criterion = criterion.strip()
            if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
                continue
            if len(criterion) < 5:
                continue
            idx2criterion[str(idx)] = criterion
            idx += 1

        for idx, info in enumerate(prediction[inc_exc].items()):
            criterion_idx, preds = info
            if criterion_idx not in idx2criterion:
                continue
            criterion = idx2criterion[criterion_idx]
            if len(preds) != 3:
                continue
            output += f"{inc_exc} criterion {idx}: {criterion}\n"
            output += f"\tPatient relevance: {preds[0]}\n"
            if len(preds[1]) > 0:
                output += f"\tEvident sentences: {preds[1]}\n"
            output += f"\tPatient eligibility: {preds[2]}\n"
    return output

def convert_pred_to_prompt(patient, pred, trial_info):
    """Construct the system and user prompt for GPT."""
    trial = f"Title: {trial_info['brief_title']}\n"
    trial += f"Target conditions: {', '.join(trial_info['diseases_list'])}\n"
    trial += f"Summary: {trial_info['brief_summary']}"

    pred = convert_criteria_pred_to_string(pred, trial_info)

    prompt = (
        "You are a helpful assistant for clinical trial recruitment. You will be given a patient note, "
        "a clinical trial, and the patient eligibility predictions for each criterion.\n"
        "Your task is to output two scores, a relevance score (R) and an eligibility score (E), "
        "between the patient and the clinical trial.\n"
        "First explain the consideration for determining patient-trial relevance. "
        "Predict the relevance score R (0~100), which represents the overall relevance between the patient and the clinical trial. "
        "R=0 denotes the patient is totally irrelevant to the clinical trial, and R=100 denotes the patient is exactly relevant to the clinical trial.\n"
        "Then explain the consideration for determining patient-trial eligibility. "
        "Predict the eligibility score E (-R~R), which represents the patient's eligibility to the clinical trial. "
        "Note that -R <= E <= R (the absolute value of eligibility cannot be higher than the relevance), "
        "where E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), "
        "E=R denotes that the patient is eligible (included by all inclusion criteria, and not excluded by any exclusion criteria), "
        "E=0 denotes the patient is neutral (i.e., no relevant information for all inclusion and exclusion criteria).\n"
        "Please output a JSON dict formatted as Dict{\"relevance_explanation\": Str, \"relevance_score_R\": Float, "
        "\"eligibility_explanation\": Str, \"eligibility_score_E\": Float}."
    )

    user_prompt = f"Here is the patient note:\n{patient}\n\nHere is the clinical trial description:\n{trial}\n\n"
    user_prompt += f"Here are the criterion-level eligibility predictions:\n{pred}\n\nPlain JSON output:"
    return prompt, user_prompt

def trialgpt_aggregation(patient, trial_results, trial_info, model):
    """Generate aggregation scores using GPT."""
    system_prompt, user_prompt = convert_pred_to_prompt(patient, trial_results, trial_info)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    result = response.choices[0].message.content.strip()
    result = result.strip("`").strip("json")
    return json.loads(result)

if __name__ == "__main__":
    # Model and data paths
    model = "clin-inquiry-agent-gpt4"  # Replace with the actual model name
    matching_results_path = "matching_results.json"  # Path to matching results
    trial_info_path = "trial_info.json"  # Path to trial information

    # Load data
    results = json.load(open(matching_results_path))
    trial2info = json.load(open(trial_info_path))

    output = {}
    for trial_id, trial_results in results.items():
        trial_info = trial2info[trial_id]
        try:
            result = trialgpt_aggregation(PATIENT_NOTE, trial_results, trial_info, model)
            output[trial_id] = result
        except Exception as e:
            print(f"Error processing trial {trial_id}: {e}")
            output[trial_id] = {"error": str(e)}

    # Save output
    output_path = "aggregation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Results saved to {output_path}")
