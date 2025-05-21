import json
from transformers import AutoTokenizer, AutoModel
import faiss
import os
from nltk import word_tokenize
from rank_bm25 import BM25Okapi
import numpy as np
import torch


corpus_file = 'storage/corpus.jsonl'

bm25_cache_file = 'storage/embeddings/bm25_cache.json'

medcpt_embed_cache = "storage/embeddings/medcpt_embeds.npy"  
medcpt_id_cache = "storage/embeddings/medcpt_doc_ids.json"



#creating bm25 tokenizer.
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

                #-------------
                for keyword in entry.get("metadata", {}).get("keywords", []):
                    tokens += word_tokenize(keyword.lower()) * 2
                #-------------

                tokens += word_tokenize(entry["text"].lower())

                tokenized_corpus.append(tokens)

        # Save cache
        with open(cache_file, "w") as f:
            json.dump({"tokenized_corpus": tokenized_corpus, "doc_ids": doc_ids, "doc_titles": doc_titles}, f, indent=4)

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

