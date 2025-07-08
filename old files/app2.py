import streamlit as st
import json
import subprocess
import re
import time

from retrieval_module import hybrid_retriever
from matching_module import matching
from ranking_module import ranking


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
new_note = st.text_area("Enter the patient info:")



if st.button("Extract Trials"):
    start_time = time.time()
    if new_note:
        try:
            update_patient_note(json_file_path, new_note)
            st.divider()
            st.header("Stage 1: Retrieval")
            with st.spinner(text="Performing Lexical and Semantic search to retrieve relevant trials..."):
                # result = subprocess.run(
                #         ["python", "retrieval.py"],
                #         capture_output=True,
                #         text=True,
                #         check=True
                #     )
                
                result = hybrid_retriever()
                # st.success("Trials fetched successfully!"
                # st.text(result.stdout)

                output_data = ""

                try:
                    output_data = json.loads(result)
                except:
                    match = re.search(r'\{.*\}', result, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        try:
                            output_data = json.loads(json_str)
                            
                            # summary = output_data.get("summary", "No summary found")
                            # st.markdown(f"The patient summary is: **{summary}**")

                            # keywords = output_data.get("conditions", "No keyword found")
                            # st.markdown(f"The keywrods generated for this patient are: {keywords}")
                        except json.JSONDecodeError:
                            st.error("Failed to parse retrieval output as JSON")
                    else:
                        st.error("Could not find JSON data in output")

                if output_data:
                    summary = output_data.get("summary", "No summary found")
                    st.markdown(f"The patient summary is: **{summary}**")

                    keywords = output_data.get("conditions", "No keyword found")
                    st.markdown(f"The keywrods generated for this patient are: **{keywords}**")



                # summary = result["summary"]
                # st.text(summary)

            #prinitng trails from retrieved_trails.json

            
            retrieved_trials_data = load_json(retrieved_trials)

            st.markdown('After **Stage 1 filetration**, we narrow down the search space to:\n\n')

            idx = 1
            for trial in retrieved_trials_data["retrieved_trials"]:
                st.text(f"{idx}. {trial}")
                idx += 1

            detailed_trials = load_json(detailed_results)

            if isinstance(detailed_trials, list):
                detailed_trials = detailed_trials[:3]

            st.subheader("Details of retrieved trials: ")
            st.json(detailed_trials)


#*****stage 2******

            #Running Matching Stage
            st.divider()
            st.header("Stage 2: Matching")
            with st.spinner(text="Performing analysis on inclusion and exclusion criterias..."):
                # result = subprocess.run(
                #         ["python", "matching.py"],
                #         capture_output=True,
                #         text=True,
                #         check=True
                #     )
                result = matching()

            matched_trials = load_json('storage/matching_results.json')
            st.markdown('After performing **Inclusion and Exclusion matching** on each clinical trial:')
            st.json(matched_trials)



#****stage 3*******

            st.divider()
            st.header("Stage 3: Ranking")

            # with st.spinner(text="Scoring each trial based upon Inlcusion and Exclusion criterias..."):
            #     try:
            #         result = subprocess.run(
            #             ["python", "aggregation.py"],
            #             capture_output=True,
            #             text=True,
            #             check=True
            #         )
                    
            #         st.text('Calculated Relevane Score and Eligibility Score for each trial')
            #         scored_trials = load_json('storage/aggregation_results.json')

            #         st.json(scored_trials)            
                
            #     except FileNotFoundError:
            #         st.error("Error with LLM output!")
            
            # st.divider()
            try:
                # result = subprocess.run(
                #     ["python", "ranking_streamlit.py"],
                #     capture_output=True,
                #     text=True,
                #     check=True
                # )

                st.subheader("Ranked Trials:")

                result = ranking()
                st.text(result)

                # st.markdown(f"{result.stdout}")

            except Exception as e:
                st.error(f"An error occurred: {e}")


        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    st.markdown(f"**The total run time = {time_elapsed : .2f} seconds.**")







