import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from tqdm import tqdm

# --- Configuration (Harmonized Paths) ---
BASE_MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
OUTPUT_PATH = Path('./output')
SNIPPETS_FILE = './evidence_snippets.parquet'
FAISS_INDEX_FILE = OUTPUT_PATH / 'dense_index.faiss'
DEVICE = 'cuda' 

# --- Component 1: BM25 Retriever (Keyword-based) ---
class BM25Retriever:
    """Performs keyword-based search using BM25Okapi."""
    def __init__(self, corpus):
        print("Initializing BM25Retriever...")
        tokenized_corpus = [doc.split(" ") for doc in tqdm(corpus, desc="Tokenizing for BM25")]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus = corpus
        print("BM25 index built.")

    def search(self, query, top_k=10):
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        # Return indices and scores for RRF
        top_n_indices = np.argsort(doc_scores)[::-1][:top_k]
        return [(self.corpus[i], doc_scores[i], i) for i in top_n_indices]

# --- Component 2: Dense Retriever (Semantic-based) ---
class DenseRetriever:
    """Performs semantic search using SentenceTransformer and FAISS."""
    def __init__(self, corpus, model_name=BASE_MODEL_NAME):
        print(f"Initializing DenseRetriever with model: {model_name} on device: {DEVICE}...")
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.corpus = corpus
        self.faiss_index_path = FAISS_INDEX_FILE
        self.index = None
        if DEVICE == 'cuda':
            print("DenseRetriever model successfully loaded to CUDA.")

    def build_index(self, force_rebuild=False):
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        if self.faiss_index_path.exists() and not force_rebuild:
            print(f"Loading existing FAISS index from {self.faiss_index_path}...")
            self.index = faiss.read_index(str(self.faiss_index_path))
            print(f"FAISS index loaded. Total vectors: {self.index.ntotal}")
        else:
            print("Building new FAISS index for dense retrieval (this may take a while)...")
            embeddings = self.model.encode(
                self.corpus,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=DEVICE
            )
            embeddings = embeddings.cpu().numpy()
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            print(f"FAISS index built. Total vectors: {self.index.ntotal}")
            faiss.write_index(self.index, str(self.faiss_index_path))
            print(f"FAISS index saved to {self.faiss_index_path}")

    def search(self, query, top_k=10):
        if self.index is None:
            raise RuntimeError("FAISS index is not built or loaded. Call `build_index()` first.")
        
        query_embedding = self.model.encode(query, convert_to_tensor=True, device=DEVICE)
        query_embedding = query_embedding.cpu().numpy().reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return indices and scores for RRF
        results = []
        for i in range(len(indices[0])):
            original_index = indices[0][i]
            if original_index == -1: continue # Skip invalid indices
            snippet = self.corpus[original_index]
            # Convert L2 distance to a similarity score (0-1). Closer to 1 is better.
            score = 1 / (1 + distances[0][i]) 
            results.append((snippet, score, original_index))
        return results

# --- Component 3: Hybrid Retriever with RRF ---
class HybridRetriever:
    """Combines BM25 and Dense retrieval using Reciprocal Rank Fusion."""
    def __init__(self, bm25_retriever, dense_retriever):
        print("Initializing HybridRetriever with RRF...")
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever

    def search(self, query, top_k=5, k=60):
        """
        Performs a hybrid search using Reciprocal Rank Fusion (RRF).
        :param k: A constant used in the RRF formula, defaults to 60.
        """
        bm25_results = self.bm25_retriever.search(query, top_k=top_k*5)
        dense_results = self.dense_retriever.search(query, top_k=top_k*5)
        
        # --- Reciprocal Rank Fusion (RRF) Implementation ---
        rrf_scores = {}
        
        # Process BM25 results
        for rank, (snippet, score, original_index) in enumerate(bm25_results):
            if original_index not in rrf_scores:
                rrf_scores[original_index] = 0
            rrf_scores[original_index] += 1 / (k + rank + 1)
            
        # Process Dense results
        for rank, (snippet, score, original_index) in enumerate(dense_results):
            if original_index not in rrf_scores:
                rrf_scores[original_index] = 0
            rrf_scores[original_index] += 1 / (k + rank + 1)
            
        # Sort results based on RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
        
        # Compile final results with snippets and scores
        final_results = []
        for idx in sorted_indices[:top_k]:
            snippet = self.dense_retriever.corpus[idx]
            score = rrf_scores[idx]
            final_results.append((snippet, score))
            
        return final_results

# --- Main Execution Block for Testing ---
if __name__ == '__main__':
    print("Starting Retrieval System Build and Test...")
    
    try:
        snippets_df = pd.read_parquet(SNIPPETS_FILE)
    except FileNotFoundError:
        print(f"FATAL: Snippets file not found at {SNIPPETS_FILE}.")
        print("Please run 'preprocessing.py' first.")
        exit()

    evidence_corpus = snippets_df['text'].tolist()

    # Initialize retrievers
    bm25 = BM25Retriever(evidence_corpus)
    dense = DenseRetriever(evidence_corpus)
    dense.build_index(force_rebuild=False) # Will load the saved FAISS index
    hybrid = HybridRetriever(bm25, dense)

    print("\n--- Testing Hybrid Retrieval System with RRF ---")
    test_queries = [
        "medication for high blood pressure",
        "recent lab results for hemoglobin",
        "patient with diabetes and anxiety"
    ]

    for query in test_queries:
        print(f"\nHybrid Search Results for: '{query}'")
        results = hybrid.search(query, top_k=5)
        for i, (snippet, score) in enumerate(results):
            print(f"  {i+1}. (RRF Score: {score:.4f}) | Snippet: {snippet}")

