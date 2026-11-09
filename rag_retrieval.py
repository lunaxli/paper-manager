import numpy as np
from fastembed import TextEmbedding
from pdf_parser import extract_pdf_pages,build_chunks


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def top_k_search(query_vector, embeddings, k=10):
    scores = []

    for i, vector in enumerate(embeddings):
        score = cosine_similarity(query_vector, vector)

        scores.append({
            "index": i,
            "score": score
        })

    ranked = sorted(
        scores,
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked[:k]


def build_context(results, all_chunks):
    context_parts = []

    for item in results:
        index = item["index"]
        chunk = all_chunks[index]

        part = (
            f"页码：{chunk['page']}，"
            f"chunk_id：{chunk['chunk_id']}\n"
            f"{chunk['text']}"
        )

        context_parts.append(part)

    return "\n\n".join(context_parts)


def build_sources(results, all_chunks):
    sources = []

    for item in results:
        index = item["index"]
        chunk = all_chunks[index]
        source = {
            "page":chunk["page"],
            "chunk_id":chunk["chunk_id"],
            "score":item["score"],
            "preview":chunk["text"][:150]
        }
        sources.append(source)
    return sources


def retrieve_context_with_sources(
    question, all_chunks, model, embeddings, k=10
):
    query_vector = list(model.embed([question]))[0]

    results = top_k_search(query_vector,embeddings,k=k)

    context = build_context(results,all_chunks)

    sources = build_sources(results,all_chunks)

    return context,sources


def prepare_retriever(pdf_path):
    pages = extract_pdf_pages(pdf_path)

    all_chunks = build_chunks(pages)

    chunk_texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    embeddings = list(model.embed(chunk_texts))

    return all_chunks,model,embeddings
