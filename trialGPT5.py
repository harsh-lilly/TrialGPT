import json
import boto3

client = boto3.client(service_name="bedrock-runtime")


def parse_criteria(criteria):
	output = ""
	criteria = criteria.split("\n")
	
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
	# prompt += "The factors that allow someone to participate in a clinical study are called inclusion criteria. And, the factors that disqualify someone from participating are called exclusion criteria."
	# prompt += "Both of these criterias are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions."
	prompt += f"You need to examine {inc_exc} criteria one-by-one and indentify the inclusion criterias that enable the patient to enroll into the clinicla trial and also identify exclusion criteria matches disqualifying the patient for the trial.\n"
	prompt += f"Basically each {inc_exc} criteria will be followed by sentence id, you need to output the list of sentence ids of the inclusion criteria that enable the patient to enroll into the trial. Similarly, you need to output the list of sentence ids of the exclusion criteria that disqualify the patient for trial.\n"



	# prompt += f"You should check the {inc_exc} criterias one-by-one and output how many inclusion criterias matches enabling the patient to enroll into the clinicla trial and also output how many exclusion criteria matches disqualifying the patient for the trial.\n"
	prompt += "Then,your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.\n"
	prompt += "First explain how the patient is relevant, then the score between 0 to 100. Where 0 represents the patient is totally irrelevant and 100 represents exaclty relevant.\n"
	prompt += "Then, explain how the patient is eligible. Predict the eligibility score E such that -R <= E <= R\n"
	prompt += "Where, E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible and E=0 denotes the patient is neutral where no relevant information is availabel. None of the above two scores can be null.\n"
	prompt += "Do not provide any extra note, only output a JSON dict formatted as: dict{str(inclusion_criteria_match): list(sentence id 1, sentence id 3, etc..), str(exclusion_criteria_match): list(sentence id 1, sentence id 3, etc..), str(relevance_explanation): str(explanation), str(relevance_score_R): int(score), str(eligibility_explanation): str(explanation), str(eligibility_score_E): int(score)} \n\n"


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
			trial_info += "Inclusion criteria:\n%s\n" % parse_criteria(trial['inclusion_criteria'])
		elif inc_exc == "exclusion":
			trial_info += "Exclusion criteria:\n%s\n" % parse_criteria(trial['exclusion_criteria'])


	inc_exc = "Inclusion and Exclusion"


	prompt = get_matching_prompt(inc_exc, trial_info, patient)

	# print(prompt)

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
	# print(output)
	try:
		results = json.loads(output)
	except Exception as e:
		
		# print("Error:", e)
		print("LLM didn't output in json format!")
		# results = output

	return results


















