# TrialGPT: AI-Powered Clinical Trial Screening

## Overview
TrialGPT is an AI-driven system designed to streamline the clinical trial screening process in the pharmaceutical manufacturing industry. By leveraging advanced retrieval and ranking methodologies, TrialGPT significantly reduces screening time, improving efficiency and accuracy.

## Features
- **Multistage Retrieval Pipeline**: Utilizes patient summaries, keyword generation, BM25, and MedCPT indexing for enhanced trial retrieval.
- **Hybrid Search Mechanism**: Combines lexical (BM25) and semantic (MedCPT) search for improved recall.
- **Ranking System**: Aggregates retrieved results and ranks them based on relevance.
- **Multilingual Helpdesk Ticket Processing**: Detects ticket language, translates if necessary, categorizes, and assigns priority levels.
- **Autogen Framework Integration**: Enables agentic AI operations for processing helpdesk tickets.


## Setup Instructions
### Prerequisites
Ensure you have the following installed:
- Python 3.10.10
- Required Python dependencies from `requirements.txt`

- Run
  
 ```sh
pip install -r requirements.txt
```


### Running the App
- To intialize the streamlit app, run

 ```sh
streamlit run finalApp.py
```

### Patient Info

Use this patient information / record to run the demo app.

 ```
 A 62-year-old man sees a neurologist for progressive memory loss and jerking movements of the lower extremities. Neurologic examination confirms severe cognitive deficits and memory dysfunction. An electroencephalogram shows generalized periodic sharp waves. Neuroimaging studies show moderately advanced cerebral atrophy. A cortical biopsy shows diffuse vacuolar changes of the gray matter with reactive astrocytosis but no inflammatory infiltration.
```


