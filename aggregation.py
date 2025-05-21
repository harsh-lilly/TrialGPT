import json
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from ollama import Client
import boto3

load_dotenv()


# client = AzureOpenAI(
#     api_version="2023-09-01-preview",
#     azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

client = boto3.client(service_name="bedrock-runtime")


# Hard-coded patient note
with open('storage/input.json', 'r') as file:
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


    prompt = "You are a helpful assistant for clinical trial recruitment. You will be given a patient note, a clinical trial, and the patient eligibility predictions for each criterion.\n"
    prompt += "Your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.\n"
    prompt += "First explain the consideration for patient-trial relevance, then the score between 0 to 100. Where 0 represents the patient is totally irrelevant and 100 represents exaclty relevant.\n"
    prompt += "Then, explain the consideration for patient-trial eligibility. Predict the eligibility score E such that -R <= E <= R\n"
    prompt += "Where, E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible and E=0 denotes the patient is neutral where no relevant information is availabel. None of the above two scores can be null.\n"
    prompt += 'Do not provide any extra note, only output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float, "eligibility_explanation": Str, "eligibility_score_E": Float}.'

    user_prompt = f"Here is the patient note:\n{patient}\n\nHere is the clinical trial description:\n{trial}\n\n"
    user_prompt += f"Here are the criterion-level eligibility predictions:\n{pred}\n\nPlain JSON output:"
    return prompt, user_prompt

def trialgpt_aggregation(patient, trial_results, trial_info, model):
    """Generate aggregation scores using GPT."""
    system_prompt, user_prompt = convert_pred_to_prompt(patient, trial_results, trial_info)
    combined_prompt = system_prompt + user_prompt


    messages = [
		{"role": "user", "content": [{"text": combined_prompt}]}
	]


    response = client.converse(
        modelId="us.amazon.nova-micro-v1:0",
        messages=messages,
        inferenceConfig={
        	"temperature": 0.0
        }
    )


    message = response["output"]["message"]["content"][0]["text"] 
    message = message.strip("`").strip("json")

    print(message)
    return json.loads(message)

if __name__ == "__main__":
    # Model and data paths
    model = "clin-inquiry-agent-gpt4"  
    matching_results_path = "storage/matching_results.json"  
    trial_info_path = "storage/dataset.json" 

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
            continue
            print(f"Error processing trial {trial_id}: {e}")
            output[trial_id] = {"error": str(e)}

    # Save output
    output_path = "storage/aggregation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Results saved to {output_path}")
