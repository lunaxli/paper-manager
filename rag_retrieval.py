# ResearchAgent：论文语义检索模块

import numpy as np
from fastembed import TextEmbedding
from pdf_parser import extract_pdf_pages,build_chunks


# ==========================================
# 1. 计算余弦相似度
# ==========================================

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


# ==========================================
# 2. Top-K 语义检索
# ==========================================

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


# ==========================================
# 3. 构建 RAG Context
# ==========================================

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


# ==========================================
# 4. 相邻 Chunk 扩展
# ==========================================

def expand_neighbors(results, all_chunks, window=2):
    expanded_indices = set()

    for item in results:
        center = item["index"]
        center_page = all_chunks[center]["page"]

        start = max(0, center - window)
        end = min(
            len(all_chunks),
            center + window + 1
        )

        for i in range(start, end):
            # 只扩展同一 PDF 页的 Chunk
            if all_chunks[i]["page"] == center_page:
                expanded_indices.add(i)

    return [
        all_chunks[i]
        for i in sorted(expanded_indices)
    ]


# ==========================================
# 6. 用户问题及语义检索
# ==========================================

def retrieve_context_with_sources(
    question, all_chunks, model, embeddings, k=10
):
    # TODO 1：将 question 转换为 query_vector
    query_vector = list(model.embed([question]))[0]

    # TODO 2：调用 top_k_search()
    # 得到 results
    results = top_k_search(query_vector,embeddings,k=k)

    # TODO 3：调用 build_context()
    # 得到 context
    context = build_context(results,all_chunks)

    # TODO 4：调用 build_sources()
    # 得到 sources
    sources = build_sources(results,all_chunks)

    # TODO 5：同时返回 context 和 sources
    return context,sources

def prepare_retriever(pdf_path):
    # TODO 1：调用 extract_pdf_pages(pdf_path)
    # 获得 pages
    pages = extract_pdf_pages(pdf_path)

    # TODO 2：调用 build_chunks(pages)
    # 获得 all_chunks
    all_chunks = build_chunks(pages)

    # TODO 3：提取所有 Chunk 的 text
    # 获得 chunk_texts
    chunk_texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    # TODO 4：初始化 TextEmbedding 模型
    # 模型仍使用 BAAI/bge-small-en-v1.5
    model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # TODO 5：调用 model.embed(chunk_texts)
    # 获得 embeddings
    embeddings = list(model.embed(chunk_texts))

    # TODO 6：返回三个对象
    # all_chunks、model、embeddings
    return all_chunks,model,embeddings


# ==========================================
# 8. 独立运行时的调试与验证
# ==========================================


if __name__ == "__main__":

    # 1. 准备论文与向量
    all_chunks, model, embeddings = prepare_retriever(
        "data/sample.pdf"
    )

    # 2. 设置测试问题
    question = (
        "How do capillary burst valves "
        "enable sequential sweat collection?"
    )

    # 3. 获取检索结果
    query_vector = list(model.embed([question]))[0]

    results = top_k_search(
        query_vector,
        embeddings,
        k=10
    )

    # 4. 构建上下文
    context = build_context(results, all_chunks)

    # 5. 你原来打印 Top-10 的代码
    # 以及 expand_neighbors() 的测试代码
    # 继续保留在这里
    sources = build_sources(results, all_chunks)

    print("来源数量：", len(sources))
    print("第一个来源：", sources[0])