__author__ = "qiao"

"""
Generate a patient summary and search keywords for clinical trials
"""

import json
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from nltk import word_tokenize
from rank_bm25 import BM25Okapi
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

load_dotenv()

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_version="2023-09-01-preview",
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

def get_keyword_generation_messages(note, max_keywords):
    """
    Prepare the messages for keyword generation.
    """
    system = (
        f'You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. '
        f'Please first summarize the main medical problems of the patient. Then generate up to {max_keywords} key conditions for searching relevant clinical trials for this patient. '
        'The key condition list should be ranked by priority. Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'
    )

    prompt = f"Here is the patient description: \n{note}\n\nJSON output:"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    return messages

def generate_summary_and_keywords(patient_note, max_keywords=32, model="clin-inquiry-agent-gpt4"):
    """
    Generate a patient summary and search keywords from the given note.
    """
    messages = get_keyword_generation_messages(patient_note, max_keywords)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    output = response.choices[0].message.content
    output = output.strip("`").strip("json")

    try:
        result = json.loads(output)
        return result
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None

def create_bm25_index(corpus_file, cache_file):
    """
    Create or load a BM25 index from the corpus.

    Parameters:
    corpus_file (str): Path to the corpus file in JSONL format.
    cache_file (str): Path to the cache file for the BM25 index.

    Returns:
    BM25Okapi: The BM25 index.
    list: List of document IDs.
    """
    # Check if cache exists
    if os.path.exists(cache_file):
        # print(f"Loading BM25 index from cache: {cache_file}")
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        tokenized_corpus = cache_data["tokenized_corpus"]
        doc_ids = cache_data["doc_ids"]
        doc_titles = cache_data["doc_titles"]
    else:
        print(f"Cache not found. Creating BM25 index from corpus: {corpus_file}")
        tokenized_corpus = []
        doc_ids = []
        doc_titles = []

        with open(corpus_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                doc_ids.append(entry["_id"])
                doc_titles.append(entry["title"])

                # Tokenize and weight fields
                tokens = word_tokenize(entry["title"].lower()) * 3
                for disease in entry.get("metadata", {}).get("diseases_list", []):
                    tokens += word_tokenize(disease.lower()) * 2
                tokens += word_tokenize(entry["text"].lower())

                tokenized_corpus.append(tokens)

        # Save cache
        with open(cache_file, "w") as f:
            json.dump({"tokenized_corpus": tokenized_corpus, "doc_ids": doc_ids, "doc_titles": doc_titles}, f, indent=4)

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, doc_ids, doc_titles

def create_medcpt_index(corpus_file, embed_cache, id_cache):
    """
    Create or load a MedCPT index from the corpus.

    Parameters:
    corpus_file (str): Path to the corpus file in JSONL format.
    embed_cache (str): Path to the cache file for the embeddings.
    id_cache (str): Path to the cache file for document IDs.

    Returns:
    faiss.IndexFlatIP: The FAISS index for embeddings.
    list: List of document IDs.
    """
    # Check if cache exists
    if os.path.exists(embed_cache) and os.path.exists(id_cache):
        # print(f"Loading MedCPT index from cache: {embed_cache} and {id_cache}")
        embeds = np.load(embed_cache)
        doc_ids = json.load(open(id_cache))
    else:
        print(f"Cache not found. Creating MedCPT index from corpus: {corpus_file}")
        embeds = []
        doc_ids = []

        model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to("mps")
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

        with open(corpus_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                doc_ids.append(entry["_id"])

                title = entry["title"]
                text = entry["text"]

                with torch.no_grad():
                    encoded = tokenizer(
                        [[title, text]],
                        truncation=True,
                        padding=True,
                        return_tensors='pt',
                        max_length=512
                    ).to("mps")

                    embed = model(**encoded).last_hidden_state[:, 0, :]
                    embeds.append(embed[0].cpu().numpy())

        embeds = np.array(embeds)

        # Save cache
        np.save(embed_cache, embeds)
        with open(id_cache, "w") as f:
            json.dump(doc_ids, f, indent=4)

    # Create FAISS index
    index = faiss.IndexFlatIP(768)
    index.add(embeds)
    return index, doc_ids

def hybrid_retrieval_and_fusion(query, bm25, bm25_doc_ids, bm25_doc_titles, medcpt_index, medcpt_doc_ids, bm25_wt=1, medcpt_wt=1, top_n=100):
    """
    Perform hybrid retrieval and fusion for a given query.

    Parameters:
    query (str): The query string.
    bm25 (BM25Okapi): The BM25 index.
    bm25_doc_ids (list): Document IDs for BM25.
    bm25_doc_titles (list): Document titles for BM25.
    medcpt_index (faiss.IndexFlatIP): The FAISS index for MedCPT embeddings.
    medcpt_doc_ids (list): Document IDs for MedCPT.
    bm25_wt (int): Weight for BM25 scores.
    medcpt_wt (int): Weight for MedCPT scores.
    top_n (int): Number of top documents to return.

    Returns:
    list: Top N documents with their IDs and titles ranked by combined score.
    """
    # BM25 retrieval
    bm25_tokens = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(bm25_tokens)
    bm25_top_indices = np.argsort(bm25_scores)[-top_n:][::-1]
    bm25_top_scores = bm25_scores[bm25_top_indices]
    bm25_top_doc_ids = [bm25_doc_ids[i] for i in bm25_top_indices]
    bm25_top_doc_titles = [bm25_doc_titles[i] for i in bm25_top_indices]

    # MedCPT retrieval
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to("mps")

    with torch.no_grad():
        encoded_query = tokenizer(
            query,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=512
        ).to("mps")
        query_embed = model(**encoded_query).last_hidden_state[:, 0, :].cpu().numpy()

    medcpt_scores, medcpt_indices = medcpt_index.search(query_embed, top_n)
    medcpt_top_doc_ids = [medcpt_doc_ids[i] for i in medcpt_indices[0]]
    medcpt_top_scores = medcpt_scores[0]

    # Combine scores
    combined_scores = {}
    doc_id_to_title = {doc_id: title for doc_id, title in zip(bm25_doc_ids, bm25_doc_titles)}

    # Add BM25 scores
    for doc_id, score in zip(bm25_top_doc_ids, bm25_top_scores):
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + bm25_wt * score

    # Add MedCPT scores
    for doc_id, score in zip(medcpt_top_doc_ids, medcpt_top_scores):
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + medcpt_wt * score

    # Rank documents by combined scores
    ranked_docs = sorted(combined_scores.items(), key=lambda x: -x[1])[:top_n]

    return [(doc_id, doc_id_to_title.get(doc_id, "Unknown Title")) for doc_id, _ in ranked_docs]

def calculate_recall(test_file, query_id, retrieved_doc_ids):
    """
    Calculate the recall for a specific query ID.

    Parameters:
    test_file (str): Path to the test file (TSV format).
    query_id (str): Query ID for which recall needs to be calculated.
    retrieved_doc_ids (list): List of document IDs retrieved for the query.

    Returns:
    float: Recall value for the query.
    """
    relevant_docs = set()
    
    # Read the test file to extract relevant documents for the given query ID
    with open(test_file, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if parts[0] == query_id and int(parts[2]) > 0:
                relevant_docs.add(parts[1])
    
    if not relevant_docs:
        print(f"No relevant documents found for query ID: {query_id}")
        return 0.0

    # Calculate recall
    retrieved_relevant_docs = [doc_id for doc_id in retrieved_doc_ids if doc_id in relevant_docs]
    recall = len(retrieved_relevant_docs) / len(relevant_docs)
    return recall


if __name__ == "__main__":
    # Example patient record

    # Open the JSON file and load it into a Python dictionary
    with open('input.json', 'r') as file:
        data = json.load(file)

    patient_note = data['patient_note']
    max_keywords = 32  # Define the maximum number of keywords

    result = generate_summary_and_keywords(patient_note, max_keywords=max_keywords)

    if result:
        print("\nPatient Summary and Keywords:\n")
        print(json.dumps(result, indent=4))
    else:
        print("Failed to generate summary and keywords.")

    # Example usage of BM25 index creation
    corpus_file = "corpus.jsonl"  # Path to the corpus file
    bm25_cache_file = "bm25_cache.json"  # Path to the BM25 cache file

    bm25_index, bm25_document_ids, bm25_document_titles = create_bm25_index(corpus_file, bm25_cache_file)
    # print("BM25 index created. Document count:", len(bm25_document_ids))

    # Example usage of MedCPT index creation
    medcpt_embed_cache = "medcpt_embeds.npy"  # Path to the embedding cache
    medcpt_id_cache = "medcpt_doc_ids.json"  # Path to the document ID cache

    medcpt_index, medcpt_document_ids = create_medcpt_index(corpus_file, medcpt_embed_cache, medcpt_id_cache)
    # print("MedCPT index created. Document count:", len(medcpt_document_ids))

    # Example hybrid retrieval
    query = result["summary"] if result else "high fever, conjunctivitis, strawberry tongue, and coronary artery dilation"
    top_docs = hybrid_retrieval_and_fusion(
        query,
        bm25_index,
        bm25_document_ids,
        bm25_document_titles,
        medcpt_index,
        medcpt_document_ids,
        bm25_wt=1,
        medcpt_wt=1,
        top_n=3
    )
    # print("\nTop documents from hybrid retrieval:")
    # for doc_id, title in top_docs
    #     print(f"ID: {doc_id}, Title: {title}")

    # Save retrieved trial IDs to a JSON file
    retrieved_trial_ids = [doc_id for doc_id, _ in top_docs]
    with open("retrieved_trials.json", "w") as f:
        json.dump({"retrieved_trials": retrieved_trial_ids}, f, indent=4)
    print("\nRetrieved Clinical Trial IDs saved to 'retrieved_trials.json'.")

    # Evaluate recall
    test_file = "test.tsv"
    query_id = data['patient_id']
    recall = calculate_recall(test_file, query_id, retrieved_trial_ids)
    # print(f"\nRecall for query ID {query_id}: {recall:.4f}")
    print(f"Recall for this Patient Note is: {recall:.4f}")
