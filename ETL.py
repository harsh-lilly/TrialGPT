import json
import re

dataset = "storage/trials_data.json"
filtered_data = "storage/dataset.json"

with open(dataset, 'r') as file:
    data = json.load(file)

# to store the result
result = {}

# List to store trials in JSONL format
trial_list = []
count = 0

for study in data.get("studies", []): # [] returns null object if key not found.

    identification = study.get("protocolSection", {}).get("identificationModule", {})

    nct_id = identification.get("nctId")
    brief_title = identification.get("briefTitle") 
    official_title = identification.get("officialTitle")

    #lilly alias
    secondary_ids = identification.get("secondaryIdInfos", [])
    secondary_alias = [entry["id"] for entry in secondary_ids if "id" in entry]

    #extracting lilly alias from this list of id's
    lilly_alias = []
    pattern = r"\b[a-zA-Z0-9]{3}-[a-zA-Z0-9]{2}-[a-zA-Z0-9]{4}\b"
    lilly_alias = [s for s in secondary_alias if re.fullmatch(pattern, s)]


    #brief summary
    briefSummary = study.get("protocolSection", {}).get("descriptionModule", {}).get("briefSummary", {})

    #Trial Status
    overallStatus = study.get("protocolSection", {}).get("statusModule", {}).get("overallStatus", {})

    #Trial Phase
    phase = study.get("protocolSection", {}).get("designModule", {}).get("phases", {})

    #Eligibility Criteria
    eligible_criteria = study.get("protocolSection", {}).get("eligibilityModule", {}).get("eligibilityCriteria", {})




    # Extract Inclusion Criteria
    inclusion_match = re.search(r"(?<=Inclusion Criteria:\n\n)(.*?)(?=\n\nExclusion Criteria:)", eligible_criteria, re.DOTALL)
    inclusion_criteria = inclusion_match.group(1).strip() if inclusion_match else " "

    #formatting
    if inclusion_criteria:
        inclusion_criteria = 'inclusion criteria: \n\n' + inclusion_criteria  
        inclusion_criteria = re.sub(r"\*\s*", " ", inclusion_criteria)


    # Extract Exclusion Criteria
    exclusion_match = re.search(r"(?<=Exclusion Criteria:\n\n)(.*)", eligible_criteria, re.DOTALL)
    exclusion_criteria = exclusion_match.group(1).strip() if exclusion_match else " "

    if exclusion_criteria:
        exclusion_criteria = 'exclusion criteria: \n\n' + exclusion_criteria
        exclusion_criteria = re.sub(r"\*\s*", "", exclusion_criteria)



    #disease List
    diseases_list = study.get("protocolSection", {}).get("conditionsModule", {}).get("conditions", [])

    #Drug list
    interventions = study.get("protocolSection", {}).get("armsInterventionsModule", {}).get("interventions", [])
    drug_list = []
    for x in interventions:
        drug_name = x.get("name", [])
        drug_list.append(drug_name)


    #enrollment
    enrollment = study.get("protocolSection", {}).get("designModule", {}).get("enrollmentInfo", {}).get("count", [])


    #keywords
    keywords = study.get("protocolSection", {}).get("conditionsModule", {}).get("keywords", [])



    
    if overallStatus == "RECRUITING" and nct_id and lilly_alias:
 
        count += 1

        result[nct_id] = {
            "brief_title" : brief_title,
            "official_title" : official_title,
            "lillyAlias": lilly_alias,
            "brief_summary": briefSummary,
            "trial_status": overallStatus,
            "phase": phase,
            "diseases_list": diseases_list,
            "drugs_list" : drug_list,
            "enrollment": enrollment,
            "inclusion_criteria" : inclusion_criteria,
            "exclusion_criteria" : exclusion_criteria,
            "keywords" : keywords,
        }

        #for creating corpus - in jsonl format
        trial_data = {
            "_id": nct_id,
            "title": brief_title,
            "text": f"Summary: {briefSummary}\n\nInclusion criteria: {inclusion_criteria}\n\nExclusion criteria: {exclusion_criteria}",
            "metadata": {
                "brief_title": brief_title,
                "phase": phase,
                "drugs_list": drug_list,
                "diseases_list": diseases_list,
                "enrollment": enrollment,
                "inclusion_criteria": inclusion_criteria,
                "exclusion_criteria": exclusion_criteria,
                "brief_summary": briefSummary,
                "keywords": keywords,
            }
        }

         
        trial_list.append(trial_data)



with open(filtered_data, "w") as file:
    json.dump(result, file, indent=4)

print(f"Total Lilly Trials actively recruiting are: {count}")
print(f'Dataset has been successfully filtered out and is saved at: {filtered_data}')


filtered_data_jsonl = "storage/corpus.jsonl"  

# Save as JSONL
with open(filtered_data_jsonl, "w") as file:
    for trial in trial_list:
        file.write(json.dumps(trial) + "\n")

print(f'Dataset is converted to corpus.jsonl required format is stored at: {filtered_data_jsonl}')