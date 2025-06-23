import json

eps = 1e-9

def get_matching_score(matching):

    try:
        included = 0
        not_inc = 0
        no_info_inc = 0

        excluded = 0
        not_exc = 0
        no_info_exc = 0

        # Count inclusion criteria
        for criteria, info in matching["inclusion"].items():
            if len(info) != 3:
                continue

            if info[2] == "included":
                included += 1
            elif info[2] == "not included":
                not_inc += 1
            elif info[2] == "not enough information":
                no_info_inc += 1
        try:
            # Count exclusion criteria
            for criteria, info in matching["exclusion"].items():
                if len(info) != 3:
                    continue

                if info[2] == "excluded":
                    excluded += 1
                elif info[2] == "not excluded":
                    not_exc += 1
                elif info[2] == "not enough information":
                    no_info_exc += 1
        except Exception as e:
            print(e)

    except Exception as e:
        # print("Error:", e)
        return None
    
    score = 0   
    score += included / (included + not_inc + no_info_inc + eps)

    if not_inc > 0:
        score -= 1

    if excluded > 0:
        score -= 1

    return score
    



def ranking():
    output = ""

    matching_results_path = "storage/matching_results.json"
    trial_info_path = "storage/dataset.json"


    matching_results = json.load(open(matching_results_path))
    trial_info = json.load(open(trial_info_path))


    trial2score = {}
    relevance_explanation = {}


    for trial_id, results in matching_results.items():

        trial_score = get_matching_score(results)

        if trial_score:

            trial2score[trial_id] = trial_score

            relevance_explanation[trial_id] = results["relevance_explanation"]



    
    sorted_trial2score = sorted(trial2score.items(), key=lambda x: -x[1])
        

    

    with open('storage/dataset.json', 'r') as file:
        data = json.load(file)

    # print("Clinical trial ranking:")
    # for trial, score in sorted_trial2score:
    #     print(f"{trial}: {score:.4f}")

    # print("\nTop 5 Suggested Clinical Trials are: \n")
    for trial, score in sorted_trial2score:
        title = trial_info[trial]["brief_title"] if trial in trial_info else "Title not found"
        lilly_alias = data[trial].get('lillyAlias', [])
        explanation = relevance_explanation[trial]
   

        output += f"Lilly ID: {lilly_alias}, \nTitle: {title}, \nConfidence Score: {score:.2f} \nRelevance Explanation: {explanation}\n\n\n"


    # print("===")

    return output









