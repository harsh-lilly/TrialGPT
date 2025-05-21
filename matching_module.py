import json
from nltk.tokenize import sent_tokenize
import os
import sys
from trialGPT5 import trialgpt_matching
from openai import AzureOpenAI
from dotenv import load_dotenv
import boto3
import time

load_dotenv()

client = boto3.client(service_name="bedrock-runtime")


def matching():

    start_time = time.time()
    # Model and file paths
    model = 'clin-inquiry-agent-gpt4'
    dataset = json.load(open("storage/detailed_trials.json"))
    output_path = "storage/matching_results.json"

    # Load or initialize output
    # if os.path.exists(output_path):
    #     output = json.load(open(output_path))
    # else:

    output = {}

    # loading info from input file.
    with open('storage/input.json', 'r') as file:
        data = json.load(file)

    patient_note = data['patient_note']
    sents = sent_tokenize(patient_note)
    sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
    sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
    patient_note = "\n".join(sents) #a list of patient note which has been sentence tokenized and given a numerical ordering to each sentence.    

    # Iterate through trials in the dataset
    for trial in dataset:
        trial_id = trial["trial_id"]

        # Skip already processed trials
        if trial_id in output:
            continue

        try:
            # Match trial with patient
            results = trialgpt_matching(trial, patient_note, model)
            output[trial_id] = results

            # Save results incrementally
            

        except Exception as e:
            print(f"Error processing trial {trial_id}: {e}")
            continue
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    elapsed_time = time.time() - start_time

    # print(f"Total time elaspsed for matching: {elapsed_time:.2f} seconds.")

    return(f"Matching results saved to {output_path}.")