import streamlit as st
import json
import subprocess
import os


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


#top Bar.
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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



# Streamlit UI
st.title("TrialGPT Demo")

# Input fields
json_file_path = 'input.json'
retrieved_results_path = 'retrieved_trials.json'
detailed_results = 'detailed_trials.json'
new_note = st.text_area("Enter the patient info:")

# Update button
if st.button("Fetch Trials"):
    if json_file_path and new_note:
        try:
            update_patient_note(json_file_path, new_note)
            

            #Running Stage of Pipleine: Retrieval
            st.divider()
            st.header("Stage 1: Retrieval")
            # st.caption("Fetch the most relevant trials based upon keywords.")
            with st.spinner(text="Fetching the most relevant trials based upon keywords..."):
                try:
                    result = subprocess.run(
                        ["python", "retrieval.py"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    st.success("Trials fetched successfully!")
                    # st.text("Output from retrieval.py:")
                    st.text(result.stdout)
                
                except FileNotFoundError:
                    st.error("retrieval.py not found. Please ensure the file is in the correct location.")
            
            #Loading Retirived Trials List
            try:
                retrieved_data = load_json(retrieved_results_path)
                st.markdown("Retrieved Trials:")
                st.caption("(Fetched only the first three trials because of time constraints!)")
                st.json(retrieved_data)
            except FileNotFoundError:
                st.error("retrieved_results.json not found. Ensure the retrieval script generates this file.")
            

            #Fetching the details of Trails
            try:
                result = subprocess.run(
                    ["python", "prepare_metadata.py"],
                    capture_output=True,
                    text=True,
                    check=True
                )

                detailed_trials = load_json(detailed_results)

                if isinstance(detailed_trials, list):
                    detailed_trials = detailed_trials[:3]

                st.subheader("Details of retrieved trials: ")
                
                st.json(detailed_trials)            
            
            except FileNotFoundError:
                st.error("detailed_trials.py not found. Please ensure the file is in the correct location.")
            

            st.divider()
            #Running Matching Stage
            st.header("Stage 2: Matching")
            with st.spinner(text="Preparing trials for next stage..."):
                try:
                    result = subprocess.run(
                        ["python", "run_matching.py"],
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    matched_trials = load_json('matching_results.json')

                    if isinstance(detailed_trials, list):
                        detailed_trials = detailed_trials[:3]

                    st.subheader("Matching Trials: ")
                    st.caption("This stage introduces tranparency.")
                    st.json(matched_trials)            
                
                except FileNotFoundError:
                    st.error("run_matching.json not found. Please ensure the file is in the correct location.")

            st.divider()
            st.header("Stage 3: Ranking")

            #Running Aggregation Process
            with st.spinner(text="Scoring each trial based upon Inlcusion and Exclusion criterias..."):
                try:
                    result = subprocess.run(
                        ["python", "run_aggregation.py"],
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    
                    st.text('Calculated Relevane Score and Eligibility Score for each trial')

                    scored_trials = load_json('aggregation_results.json')

                    st.json(scored_trials)            
                
                except FileNotFoundError:
                    st.error("aggregation_results.json not found. Please ensure the file is in the correct location.")



            #end results
            st.divider()
            try:
                result = subprocess.run(
                    ["python", "results.py"],
                    capture_output=True,
                    text=True,
                    check=True
                )
               
                st.subheader("Ranked Trials:")
                st.text(result.stdout)
            
            except FileNotFoundError:
                st.error("retrieval.py not found. Please ensure the file is in the correct location.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

        #deleting file 'matching_results.json'
        try:
            os.remove('matching_results.json')
        except FileNotFoundError:
            st.error("matching_results.json not found.")



    else:
        st.warning("Please provide both the file path and the new note.")
