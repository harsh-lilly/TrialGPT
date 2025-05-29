import json
import boto3
import re

client = boto3.client(service_name="bedrock-runtime")


def parse_criteria(criteria):
	output = ""
	criteria = criteria.split("\n")
	
	idx = 1
	for criterion in criteria:
		criterion = criterion.strip()

		if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
			continue

		if len(criterion) < 25: # Skip very short criteria
			# print(f"Skipping short criterion: {criterion}")
			continue
	
		output += f"{idx}. {criterion}\n" 
		idx += 1
	return output


def get_matching_prompt(inc_exc, trial_info, patient):

	prompt = f"You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the {inc_exc} criteria of a clinical trial to determine the patient's eligibility at the criterion level.\n"
	prompt += "The factors that allow someone to participate in a clinical study are called inclusion criteria. And, the factors that disqualify someone from participating are called exclusion criteria."
	prompt += "Both of these criterias are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions."
	prompt += f"You need to examine {inc_exc} criteria one-by-one and indentify the inclusion criterias that enable the patient to enroll into the clinicla trial and also identify exclusion criteria matches disqualifying the patient for the trial.\n"
	prompt += f"Basically each {inc_exc} criteria will be followed by sentence id, you need to output the list of sentence ids of the inclusion criteria that enable the patient to enroll into the trial. Similarly, you need to output the list of sentence ids of the exclusion criteria that disqualify the patient for trial.\n"



	# prompt += f"You should check the {inc_exc} criterias one-by-one and output how many inclusion criterias matches enabling the patient to enroll into the clinicla trial and also output how many exclusion criteria matches disqualifying the patient for the trial.\n"
	prompt += "Then,your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.\n"
	prompt += "First explain how the patient is relevant, then the score between 0 to 100. Where 0 represents the patient is totally irrelevant and 100 represents exaclty relevant.\n"
	prompt += "Then, explain how the patient is eligible. Predict the eligibility score E such that -R <= E <= R\n"
	prompt += "Where, E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible and E=0 denotes the patient is neutral where no relevant information is availabel. None of the above two scores can be null.\n"
	prompt += "Do not provide any extra note, only output a JSON dict formatted as: dict{str(inclusion_criteria_match): [1,3,4,...etc], str(exclusion_criteria_match): [1,3,4,...etc], str(relevance_explanation): str(explanation), str(relevance_score_R): int(score), str(eligibility_explanation): str(explanation), str(eligibility_score_E): int(score)} \n\n"


	prompt += f"This is the clinical trial information:\n{trial_info}"

	prompt += f"Here is the patient information:\n{patient}\n\n" 
	return prompt


def converting_to_list(criterias):

	if isinstance(criterias, list):
		return criterias
	elif isinstance(criterias, str):
		criterias = criterias.split("\n")
		return [c.strip() for c in criterias if c.strip()]
	else:
		return []



def trialgpt_matching(trial: dict, patient: str, model: str):
	results = {}

	trial_info = f"Title: {trial['brief_title']}\n"
	trial_info += f"Target diseases: {', '.join(trial['diseases_list'])}\n"
	trial_info += f"Interventions: {', '.join(trial['drugs_list'])}\n"
	trial_info += f"Summary: {trial['brief_summary']}\n\n"

	inclusion_criterias = parse_criteria(trial['inclusion_criteria'])
	exclusion_criterias = parse_criteria(trial['exclusion_criteria'])
	
	
	for inc_exc in ["inclusion", "exclusion"]:


		if inc_exc == "inclusion":
			trial_info += "Inclusion criterias of the Clinical Trial:\n%s\n" % inclusion_criterias
		elif inc_exc == "exclusion":
			trial_info += "Exclusion criterias of the Clinical Trial:\n%s\n" % exclusion_criterias


	inc_exc = "Inclusion and Exclusion"


	prompt = get_matching_prompt(inc_exc, trial_info, patient)

	print(prompt)

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
	except Exception as e:
		match = re.search(r'\{.*\}', output, re.DOTALL)
		if match:
			json_str = match.group(0)
			try:
				results = json.loads(json_str)
			
			except json.JSONDecodeError:
				print("Failed to parse retrieval output as JSON")
		else:
			print("Could not find JSON data in output")


		# print("Error:", e)
		# print("LLM didn't output in json format!")
		# results = output

	#see the output	
	# print(results)

	# try:

	inclusion_criterias_list = (converting_to_list(inclusion_criterias))
	exclusion_criterias_list = (converting_to_list(exclusion_criterias))

	list_of_criteria = {
		"inclusion_criteria": inclusion_criterias_list,
		"exclusion_criteria": exclusion_criterias_list
	}

	list_of_inclusion = []
	list_of_exclusion = []

	try:
		for idx in results.get("inclusion_criteria_match", []):
			try:
				i = int(idx) - 1
				if 0 <= i < len(inclusion_criterias_list):
					list_of_inclusion.append(inclusion_criterias_list[i])
			except Exception:
				continue

		for idx in results.get("exclusion_criteria_match", []):
			try:
				i = int(idx) - 1
				if 0 <= i < len(exclusion_criterias_list):
					list_of_exclusion.append(exclusion_criterias_list[i])
			except Exception:
				continue

		if not list_of_inclusion:
			list_of_inclusion = ["No inclusion criteria matched."]
		if not list_of_exclusion:
			list_of_exclusion = ["No exclusion criteria matched."]

		results['list_of_inclusion'] = list_of_inclusion
		results['list_of_exclusion'] = list_of_exclusion

		# print(list_of_inclusion)
		# print(list_of_exclusion)
		

	except Exception as e:
		print("Error in finding the criterias:", e)


	return results, list_of_criteria


















