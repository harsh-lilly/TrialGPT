import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from nltk.tokenize import sent_tokenize
import os
import sys
from TrialGPT import trialgpt_matching
from openai import AzureOpenAI
from dotenv import load_dotenv
import boto3
import time

#does mathcing asynchronously

def matching():
    start_time = time.time()
    model = 'clin-inquiry-agent-gpt4'
    dataset = json.load(open("storage/detailed_trials.json"))
    output_path = "storage/matching_results.json"

    if os.path.exists(output_path):
        os.remove(output_path)

    output = {}

    with open('storage/input.json', 'r') as file:
        data = json.load(file)

    patient_note = data['patient_note']
    sents = sent_tokenize(patient_note)
    sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
    sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents, start=1)]
    patient_note = "\n".join(sents)

    async def process_trial(trial, patient_note, model, loop, executor):
        trial_id = trial["trial_id"]
        try:
            results = await loop.run_in_executor(executor, trialgpt_matching, trial, patient_note, model)
            return trial_id, results
        except Exception as e:
            print(f"Error processing trial {trial_id}: {e}")
            return trial_id, None

    async def main():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
            tasks = [
                process_trial(trial, patient_note, model, loop, executor)
                for trial in dataset
            ]
            for future in asyncio.as_completed(tasks):
                trial_id, results = await future
                if results is not None:
                    output[trial_id] = results

        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

        elapsed_time = time.time() - start_time
        print(f"Matching results saved to {output_path}.")
        print(f"Total time elapsed for matching: {elapsed_time:.2f} seconds.")

    asyncio.run(main())