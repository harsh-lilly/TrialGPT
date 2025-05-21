import json

eps = 1e-9

def get_matching_score(matching):
    """Calculate the matching score based on inclusion and exclusion criteria."""
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


    # Compute the matching score
    score = 0   
    score += included / (included + not_inc + no_info_inc + eps)

    if not_inc > 0:
        score -= 1

    if excluded > 0:
        score -= 1

    return score


def get_agg_score(assessment):
    """Calculate the aggregation score based on relevance and eligibility scores."""
    try:
        rel_score = float(assessment["relevance_score_R"])
        eli_score = float(assessment["eligibility_score_E"])
    except (KeyError, ValueError):
        rel_score = 0
        eli_score = 0

    score = (rel_score + eli_score) / 100
    return score


if __name__ == "__main__":
    # File paths
    matching_results_path = "storage/matching_results.json"
    agg_results_path = "storage/aggregation_results.json"
    trial_info_path = "storage/dataset.json"

    # Load results
    matching_results = json.load(open(matching_results_path))
    agg_results = json.load(open(agg_results_path))
    trial_info = json.load(open(trial_info_path))


    # Loop over trials
    trial2score = {}

    for trial_id, results in matching_results.items():
        matching_score = get_matching_score(results)

        # Check for aggregation results
        if trial_id not in agg_results:
            print(f"Trial {trial_id} not in the aggregation results.")
            agg_score = 0
        else:
            agg_score = get_agg_score(agg_results[trial_id])

        # Calculate final trial score
        trial_score = matching_score + agg_score
        trial2score[trial_id] = trial_score

    # Sort trials by score in descending order
    sorted_trial2score = sorted(trial2score.items(), key=lambda x: -x[1])

    with open('storage/dataset.json', 'r') as file:
        data = json.load(file)



    # Display ranked results
    print("Clinical trial ranking:")
    for trial, score in sorted_trial2score:
        print(f"{trial}: {score:.4f}")

    print("\nTop 5 Clinical Trials:")
    for trial, score in sorted_trial2score[:5]:
        title = trial_info[trial]["brief_title"] if trial in trial_info else "Title not found"
        lilly_alias = data[trial].get('lillyAlias', [])
        print(f"Lilly ID: {lilly_alias}, Trial ID: {trial}, Title: {title}, Score: {score:.4f}")


    print("===")
