import json
from nltk.tokenize import sent_tokenize
import time
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from ollama import Client
import boto3
import re

client = boto3.client(service_name="bedrock-runtime")


def parse_criteria(criteria):
	output = ""
	criteria = criteria.split("\n\n")
	
	idx = 0
	for criterion in criteria:
		criterion = criterion.strip()

		if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
			continue

		if len(criterion) < 5:
			continue
	
		output += f"{idx}. {criterion}\n" 
		idx += 1
	return output


def get_matching_prompt(inc_exc, trial_info, patient):

	prompt = f"You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the {inc_exc} criteria of a clinical trial to determine the patient's eligibility at the criterion level.\n"
	prompt += "The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"
	prompt += "And, the factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"
	prompt += f"You should check the {inc_exc} criterias one-by-one and output how many inclusion criterias matches enabling the patient to enroll into the clinicla trial and also output how many exclusion criteria matches disqualifying the patient for the trial.\n"
	prompt += "Also, you need to provide two brief explanation: first, as to how the patient is suitable for the clincial trial after considering the inclusion criterias and second, as to how the patient is unfit after considering the exclusion criterias.\n"
	prompt += "You should output only a JSON dict exactly formatted as: dict{str(inclusion_match): int(number_of_inclusion_matching), str(inclusion_explanation): str(brief_reasoning_if_patient_match_inclusion_criteria),  str(exclusion_match): int(number_of_exclusion_matching), str(exclusion_explanation): str(brief_reasoning_if_patient_match_exclusion_criteria)}. Do not output anything else.\n\n"


	prompt += f"This is the clinical trial information:\n{trial_info}"

	prompt += f"Here is the patient note, each sentence is led by a sentence_id:\n{patient}\n\n" 
	return prompt



def trialgpt_matching(trial: dict, patient: str, model: str):
	results = {}

	trial_info = f"Title: {trial['brief_title']}\n"
	trial_info += f"Target diseases: {', '.join(trial['diseases_list'])}\n"
	trial_info += f"Interventions: {', '.join(trial['drugs_list'])}\n"
	trial_info += f"Summary: {trial['brief_summary']}\n\n"
	
	
	for inc_exc in ["inclusion", "exclusion"]:

		if inc_exc == "inclusion":
			trial_info += "Inclusion criteria:\n %s\n" % parse_criteria(trial['inclusion_criteria'])
		elif inc_exc == "exclusion":
			trial_info += "Exclusion criteria:\n %s\n" % parse_criteria(trial['exclusion_criteria'])


	inc_exc = "Inclusion and Exclusion"

	prompt = get_matching_prompt(inc_exc, trial_info, patient)

	messages = [
		{"role": "user", "content": [{"text": prompt}]}
	]

	response = client.converse(
		modelId="us.amazon.nova-micro-v1:0",
		messages=messages,
		inferenceConfig={
			"temperature": 0.0
		}
	)

	output = response["output"]["message"]["content"][0]["text"] 

	try:
		results = json.loads(output)
	except:
		print("LLM didn't output in json format!")
		results = output

	return results


















