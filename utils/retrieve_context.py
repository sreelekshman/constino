from sentence_transformers import SentenceTransformer, util
import json
from collections import defaultdict
import re

def retrieve_context(query, part_threshold=0.75, max_parts=2, max_chunks=20):
    # Load the embedding model
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Load the hierarchical rag_chunks
    with open("rag_chunks_hierarchical.json", "r") as f:
        rag_chunks = json.load(f)

    selected_chunks = []
    processed_indices = set() # Tracks indices of chunks already added to selected_chunks

    # 1. Direct Article Retrieval
    # Attempts to find chunks matching specific article numbers mentioned in the query.
    queried_article_numbers = set()
    # Regex to find patterns like "Article X", "Articles X, Y and Z", "article(s) X"
    # It captures the string of numbers and separators (e.g., "15", "15 and 16", "15, 16, 17")
    article_query_match = re.search(r"article[s]?\s+((?:\d+\s*(?:(?:and|,)\s*)?)+)", query, re.IGNORECASE)
    
    if article_query_match:
        numbers_str = article_query_match.group(1)
        # Extract all individual numbers from the captured string
        found_numbers = re.findall(r'\d+', numbers_str)
        queried_article_numbers.update(found_numbers)

    if queried_article_numbers:
        for i, chunk_data in enumerate(rag_chunks):
            if len(selected_chunks) >= max_chunks:
                break 
            
            chunk_article_val = chunk_data.get("article")
            # Check if the chunk's article field matches one of the queried numbers
            if chunk_article_val is not None and str(chunk_article_val) in queried_article_numbers:
                if i not in processed_indices: # Ensure chunk hasn't been added
                    selected_chunks.append({
                        "part": chunk_data.get("part"),
                        "chapter": chunk_data.get("chapter"),
                        "article": chunk_article_val, # Store the original article value
                        "clause": chunk_data.get("clause"),
                        "text": chunk_data.get("text", ""),
                        "score": 2.0,  # Assign a very high score to prioritize direct matches
                        "source": "direct_article_match"
                    })
                    processed_indices.add(i)
        
        # If direct article matches alone are enough to fill max_chunks,
        # we can sort and return them.
        if len(selected_chunks) >= max_chunks:
            selected_chunks.sort(key=lambda x: x["score"], reverse=True)
            return selected_chunks[:max_chunks]

    # If not enough chunks from direct matches, proceed with embedding-based methods.
    # Encode query and all chunk texts (once, for subsequent semantic matching)
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    # Extract texts for all chunks for embedding
    chunk_texts = [chunk.get("text", "") for chunk in rag_chunks]
    chunk_embeddings = embed_model.encode(chunk_texts, convert_to_tensor=True)

    # 2. Part-Level Prioritization
    if len(selected_chunks) < max_chunks:
        part_chunks_map = defaultdict(list)
        for i, chunk_data in enumerate(rag_chunks):
            part = chunk_data.get("part")
            if part:
                part_chunks_map[part].append((i, chunk_data)) # Store original index and chunk
        
        part_texts_representation = {}
        if part_chunks_map:
            for part, chunks_in_part_list in part_chunks_map.items():
                # Create a representative text for each part using snippets from its initial chunks
                part_texts_representation[part] = " ".join([
                    c_data_tuple[1].get("text", "")[:100] 
                    for c_data_tuple in chunks_in_part_list[:min(5, len(chunks_in_part_list))]
                ])

        if part_texts_representation:
            part_embeddings = embed_model.encode(list(part_texts_representation.values()), convert_to_tensor=True)
            part_scores_tensor = util.cos_sim(query_embedding, part_embeddings)[0]
            part_names_list = list(part_texts_representation.keys())
            
            high_scoring_parts_info = []
            for i, score_val in enumerate(part_scores_tensor):
                if score_val.item() >= part_threshold:
                    high_scoring_parts_info.append((part_names_list[i], score_val.item()))
            
            high_scoring_parts_info.sort(key=lambda x: x[1], reverse=True)
            # Consider chunks from up to 'max_parts' highest scoring parts
            actual_high_scoring_parts_to_process = high_scoring_parts_info[:max_parts] 
            
            for part_name, part_score in actual_high_scoring_parts_to_process:
                if len(selected_chunks) >= max_chunks:
                    break
                # Iterate over chunks belonging to this high-scoring part
                for chunk_idx, chunk_data in part_chunks_map[part_name]:
                    if len(selected_chunks) >= max_chunks:
                        break
                    if chunk_idx not in processed_indices:
                        selected_chunks.append({
                            "part": chunk_data.get("part"),
                            "chapter": chunk_data.get("chapter"),
                            "article": chunk_data.get("article"),
                            "clause": chunk_data.get("clause"),
                            "text": chunk_data.get("text", ""),
                            "score": part_score, # Use the part's score for all its chunks
                            "source": "part_match"
                        })
                        processed_indices.add(chunk_idx)

    # 3. Individual Chunk Matching (if still need more chunks)
    if len(selected_chunks) < max_chunks:
        individual_chunk_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

        num_remaining_needed = max_chunks - len(selected_chunks)
        if num_remaining_needed > 0:
            # Determine how many top chunks to retrieve. Fetch a bit more to account for filtering.
            top_k_val = min(max(num_remaining_needed + 10, 1), len(individual_chunk_scores))

            if top_k_val > 0: # Ensure k is positive
                # use util.semantic_search for top-k retrieval
                hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=top_k_val)[0]
                for hit in hits:
                    if len(selected_chunks) >= max_chunks:
                        break
                    chunk_idx = hit['corpus_id']
                    score = hit['score']
                    if chunk_idx not in processed_indices:
                        chunk_data = rag_chunks[chunk_idx]
                        selected_chunks.append({
                            "part": chunk_data.get("part"),
                            "chapter": chunk_data.get("chapter"),
                            "article": chunk_data.get("article"),
                            "clause": chunk_data.get("clause"),
                            "text": chunk_data.get("text", ""),
                            "score": score,
                            "source": "individual_match"
                        })
                        processed_indices.add(chunk_idx)
    
    # Final sort of all collected chunks by score
    selected_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    return selected_chunks[:max_chunks]