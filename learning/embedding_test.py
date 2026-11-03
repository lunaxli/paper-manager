#10-6：第一次生成文本向量
#10-7｜利用余弦相似度比较论文内容

from fastembed import TextEmbedding
import numpy as np

texts = [
    "The wearable SERS device monitors lactate, glucose, urea and pH in sweat.",
    "The microfluidic system uses capillary burst valves to collect sweat sequentially.",
    "SHAP analysis identifies lactate accumulation and glucose depletion as key indicators of metabolic fatigue."
]

model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
# TODO 1：
# 使用 model.embed(texts) 生成向量
# 再使用 list() 将结果转换成列表
embeddings = list(model.embed(texts))

# TODO 2：
# 打印生成了多少个向量
# 打印第一个向量的维度
# print("向量总数:",len(embeddings))
# print("第一个向量维度",len(embeddings[0]))

question = "How does the device collect sweat sequentially?"
query_vector = list(model.embed([question]))[0]


def cosine_similarity(a, b):
    # TODO 1：使用 np.dot() 计算点积
    dot = np.dot(a,b)

    # TODO 2：使用 np.linalg.norm() 
    # 分别计算两个向量的长度
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # TODO 3：根据余弦相似度公式返回结果
    return dot/(norm_a*norm_b)
# for i, vector in enumerate(embeddings):
#     score = cosine_similarity(query_vector, vector)
#     print(f"文本{i + 1}：相似度 {score:.4f}")

#10-8｜实现 Top-K 语义检索

def top_k_search(query_vector, embeddings, k=2):
    scores = []

    for i, vector in enumerate(embeddings):
        # TODO 1：计算 query_vector 与 vector 的相似度
        score = cosine_similarity(query_vector, vector)

        # TODO 2：将 {"index": i, "score": score}
        # 添加到 scores 列表
        scores.append({"index":i,"score":float(score)})
    # TODO 3：按照 score 从高到低排序
    ranked = sorted(
        scores,
        key = lambda x:x["score"],
        reverse = True
    )

    # TODO 4：只返回前 k 个结果
    return ranked[:k]
results = top_k_search(query_vector, embeddings, k=2)

for item in results:
    print(item)