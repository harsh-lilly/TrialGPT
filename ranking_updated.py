import json


def get_matching_score(matching):

    total_inc = len(matching["inclusion_criteria_match"])
    total_exc = len(matching["exclusion_criteria_match"])

    net_criteria_score = total_inc - total_exc

    relevance_score = matching["relevance_score_R"]
    eligibility_score = matching["eligibility_score_E"]

    score = (relevance_score + eligibility_score) / 100

    return net_criteria_score + score


if __name__ == "__main__":

    matching_results_path = "storage/matching_results.json"
    trial_info_path = "storage/dataset.json"


    matching_results = json.load(open(matching_results_path))
    trial_info = json.load(open(trial_info_path))


    trial2score = {}
    relevance_explanation = {}


    for trial_id, results in matching_results.items():

        trial_score = get_matching_score(results)

        trial2score[trial_id] = trial_score

        relevance_explanation[trial_id] = results["relevance_explanation"]



    
    sorted_trial2score = sorted(trial2score.items(), key=lambda x: -x[1])
        

    

    with open('storage/dataset.json', 'r') as file:
        data = json.load(file)

    # print("Clinical trial ranking:")
    # for trial, score in sorted_trial2score:
    #     print(f"{trial}: {score:.4f}")

    print("\nTop 5 Suggested Clinical Trials are: \n")
    for trial, score in sorted_trial2score[:5]:
        title = trial_info[trial]["brief_title"] if trial in trial_info else "Title not found"
        lilly_alias = data[trial].get('lillyAlias', [])
        explanation = relevance_explanation[trial]
        print(f"Lilly ID: {lilly_alias}, \nTitle: {title}, \nConfidence Score: {score:.2f} \nRelevance Explanation: {explanation}\n\n\n")


    print("===")









