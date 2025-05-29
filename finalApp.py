import streamlit as st
import json
import re
import time
import os
from retrieval_module import hybrid_retriever
from matching_module import matching
from ranking_module import ranking
import torch
import textract
import tempfile


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Function to load the JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to update the patient_note in the JSON file
def update_patient_note(file_path, new_note):
    with open(file_path, 'r+') as file:
        data = json.load(file)
        data['patient_note'] = new_note
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()

torch.classes.__path__ = []




# Apply CSS
load_css("style.css")

# Full-width Header
st.markdown("""
    <div class="header-container">
        <h1 class="header-text">Lilly</h1>
    </div>
""", unsafe_allow_html=True)


# Add space to prevent content from being hidden under the fixed header
st.write("<br><br><br>", unsafe_allow_html=True)

st.title("TrialGPT Demo")

json_file_path = 'storage/input.json'
retrieved_trials = 'storage/retrieved_trials.json'
detailed_results = 'storage/detailed_trials.json'


input_method = st.radio("Choose input method:", ["Text Input (Patient Summary)", "PDF Input (Patient Report)"])

new_note = ""

if input_method == "Text Input (Patient Summary)":
    new_note = st.text_area("Enter the patient info:", height=200)

elif input_method == "PDF Input (Patient Report)":
    uploaded_file = st.file_uploader("Upload the patient report", type=["pdf"])
    if uploaded_file:
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
                
            text = textract.process(tmp_file_path, method='tesseract', language='eng')
            new_note = text.decode("utf-8")


if st.button("Extract Trials"):
    start_time = time.time()
    if new_note:
        try:
            update_patient_note(json_file_path, new_note)
            st.divider()
            st.subheader("Extracting Trials")
            with st.spinner(text="Analyzing Patient Information to extract meaningful trials..."):
                
                result = hybrid_retriever()

                output_data = ""

                try:
                    output_data = json.loads(result)
                except:
                    match = re.search(r'\{.*\}', result, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        try:
                            output_data = json.loads(json_str)
                            
                            summary = output_data.get("summary", "No summary found")
                            # st.markdown(f"The patient summary is: **{summary}**")

                            keywords = output_data.get("conditions", "No keyword found")
                            # st.markdown(f"The keywrods generated for this patient are: {keywords}")
                        except json.JSONDecodeError:
                            st.error("Failed to parse retrieval output as JSON")
                    else:
                        st.error("Could not find JSON data in output")

                if output_data:
                    summary = output_data.get("summary", "No summary found")
                    st.markdown(f"A brief summary of patient's condition: **{summary}**")

                    # keywords = output_data.get("conditions", "No keyword found")
                    # st.markdown(f"The keywrods generated for this patient are: **{keywords}**")




#*****stage 2******

            #Running Matching Stage
            # st.divider()
            # st.header("Stage 2: Matching")
            with st.spinner(text="Performing in-depth analysis of patient information with the inclusion and exclusion criterias of trials..."):
                result = matching()




#****stage 3*******

            # st.divider()
            # st.header("Stage 3: Ranking")

            try:
  
                st.subheader("Ranked Trials:")

                def get_matching_score(matching):

                    try:

                        total_inc = len(matching["inclusion_criteria_match"])
                        total_exc = len(matching["exclusion_criteria_match"])

                        net_criteria_score = total_inc - total_exc

                        relevance_score = matching["relevance_score_R"]
                        eligibility_score = matching["eligibility_score_E"]

                        score = (relevance_score + eligibility_score) / 100

                        return net_criteria_score + score

                    except Exception as e:
                        print("Error:", e)
                        return None
                                

                matching_results_path = "storage/matching_results.json"
                trial_info_path = "storage/dataset.json"


                matching_results = json.load(open(matching_results_path))
                trial_info = json.load(open(trial_info_path))


                trial2score = {}
                to_display = {}


                for trial_id, results in matching_results.items():

                    trial_score = get_matching_score(results)

                    if trial_score:
                        trial2score[trial_id] = trial_score
                        # to_display[relevance_explanation] = results["relevance_explanation"]
                        # relevance_explanation[trial_id] = results["relevance_explanation"]

                        to_display[trial_id] = {
                            "relevance_explanation": results["relevance_explanation"],
                            "list_of_inclusion": results["list_of_inclusion"],
                            "list_of_exclusion": results["list_of_exclusion"]
                        }
                        

                sorted_trial2score = sorted(trial2score.items(), key=lambda x: -x[1])
            
                with open('storage/dataset.json', 'r') as file:
                    data = json.load(file)
      
                index = 1

                for trial, score in sorted_trial2score:
                    title = trial_info[trial]["brief_title"] if trial in trial_info else "Title not found"

                    #extract the code
                    lilly_alias = data[trial].get('lillyAlias', [])
                    match = re.search(r"'([^']+)'", str(lilly_alias))

                    if match:
                        code = match.group(1)
                        lilly_alias = code

                    summary = data[trial].get('brief_summary', [])

                    explanation = to_display[trial]["relevance_explanation"]
                    inclusion_list = to_display[trial]['list_of_inclusion']
                    exclusion_list = to_display[trial]['list_of_exclusion']

                    st.text("\n\n\n\n\n")
                    st.markdown(f"{index}. **Lilly ID: {lilly_alias}**, ")
                    st.markdown(f"\n**Title:** {title},")
                    st.markdown(f"\n**Summary:** {summary[:230]}...")  # Display first 100 characters of summary
                    st.markdown(f"\n**Confidence Score:** {score:.2f},")
                    st.markdown(f"\n**Relevance Explanation:** {explanation}")
                    # st.text(f"\nInclusion Criteria Matched: {inclusion_list}")
                    st.text("\n\n")
                    st.markdown("\n\nThe **Inclusion Criteria** that match this patient are:")
                    for inc in inclusion_list:
                        st.text(f"✅ {inc}")
                    st.text("\n\n")
                    st.markdown("\n\nThe **Exclusion Criteria** that exclude this patient are:")
                    for exc in exclusion_list:
                        st.text(f"❌ {exc}")


                    # st.text(f"\nExclusion Criteria Matched: {', '.join(exclusion_list)}")

                    st.divider()
                    index += 1

                # result = ranking()
                # st.text(result)


            except Exception as e:
                st.error(f"An error occurred: {e}")


        except Exception as e:
            st.error(f"An error occurred: {e}")

    # if os.path.exists("storage/matching_results.json"):
    #     os.remove("storage/matching_results.json")

    
    end_time = time.time()
    time_elapsed = end_time - start_time
    st.markdown(f"**The total run time = {time_elapsed : .2f} seconds.**")







