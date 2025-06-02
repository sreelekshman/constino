import re
import json
import tiktoken

MAX_TOKENS = 512
ENCODING = "cl100k_base"

# Load the tokenizer
tokenizer = tiktoken.get_encoding(ENCODING)

def count_tokens(text):
    return len(tokenizer.encode(text))

# First, let's check what's in the file
with open("Constitution_of_India.txt", 'r', encoding='utf-8') as f:
    constitution_text = f.read()

def chunk_constitution(constitution_text):
    chunks = []
    parts = re.split(r'(\*?\*?PART\s+[IVXLCDM]+\*?\*?).*?\n', constitution_text, flags=re.DOTALL)
    print(f"Found {len(parts)//2} potential parts")
    current_part = None
    for i, part_text in enumerate(parts):
        if i % 2 == 1:
            current_part = part_text.strip('*')
            print(f"Processing PART: {current_part}")
            continue
        if not part_text.strip():
            continue
        chapters = re.split(r'(\*?\*?CHAPTER\s+[IVXLCDM]+\*?\*?).*?\n', part_text, flags=re.DOTALL)
        print(f"  Found {len(chapters)//2} potential chapters in part {current_part}")
        current_chapter = None
        for j, chapter_text in enumerate(chapters):
            if j % 2 == 1:
                current_chapter = chapter_text.strip('*')
                print(f"  Processing CHAPTER: {current_chapter}")
                continue
            if not chapter_text.strip():
                continue
            articles = re.split(r'(\*?\*?[Aa]rticle\s+\d+[A-Z]?\*?\*?).*?\n', chapter_text, flags=re.DOTALL)
            print(f"    Found {len(articles)//2} potential articles in chapter {current_chapter}")
            current_article = None
            for k, article_text in enumerate(articles):
                if k % 2 == 1:
                    current_article = article_text.replace('**Article ', '').replace('**article ', '')
                    current_article = current_article.replace('Article ', '').replace('article ', '')
                    current_article = current_article.replace('**', '').strip()
                    print(f"    Processing ARTICLE: {current_article}")
                    continue
                if not article_text.strip():
                    continue
                print(f"      Article text length: {len(article_text)}")
                if '(' in article_text and ')' in article_text:
                    clauses = re.split(r'\((\d+[A-Z]?)\)\s', article_text)
                else:
                    clauses = [article_text]
                current_clause = None
                curr_chunk = ""
                for l, clause_text in enumerate(clauses):
                    if len(clauses) > 1 and l % 2 == 1:
                        current_clause = clause_text
                        continue
                    if not clause_text.strip():
                        continue
                    candidate_chunk = (curr_chunk + " " + clause_text).strip() if curr_chunk else clause_text.strip()
                    token_count = count_tokens(candidate_chunk) if candidate_chunk else 0
                    if token_count <= MAX_TOKENS:
                        curr_chunk = candidate_chunk
                    else:
                        if curr_chunk and current_article is not None:
                            chunks.append({
                                "part": current_part,
                                "chapter": current_chapter,
                                "article": current_article,
                                "clause": current_clause,
                                "text": curr_chunk
                            })
                            print(f"        Added chunk: {len(chunks)}")
                        curr_chunk = clause_text.strip()
                        if curr_chunk:
                            current_chunk_token_count = count_tokens(curr_chunk)
                            if current_chunk_token_count > MAX_TOKENS:
                                print(f"Warning: Single clause/segment '{curr_chunk[:50]}...' is too long ({current_chunk_token_count} tokens) and will be added as is.")
                if curr_chunk and current_article is not None:
                    chunks.append({
                        "part": current_part,
                        "chapter": current_chapter,
                        "article": current_article,
                        "clause": current_clause,
                        "text": curr_chunk
                    })
                    print(f"        Added chunk: {len(chunks)}")
    return chunks

# Chunk the text
rag_chunks = chunk_constitution(constitution_text)

# Save the chunks to a JSON file
with open("rag_chunks_hierarchical.json", "w", encoding='utf-8') as f:
    json.dump(rag_chunks, f, indent=4)

print(f"Generated {len(rag_chunks)} RAG chunks with hierarchical structure.")