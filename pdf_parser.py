#读取PDF，生成pages
import pymupdf

def extract_pdf_pages(pdf_path):
    pages = []

    doc = pymupdf.open(pdf_path)

    for page_num,page in enumerate(doc,start=1):
        item = {
            "page":page_num,
            "text":page.get_text()
        }
        pages.append(item)
    
    doc.close()
    return pages

# #10-3｜将论文切分成多个文本块
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap需要满足0 <= overlap < chunk_size")

    while start < len(text):
        # TODO 1：计算 end
        end = min(start+chunk_size,len(text))

        # TODO 2：使用 text[start:end] 提取文本
        chunk = text[start:end]

        # TODO 3：将文本块添加到 chunks
        chunks.append(chunk)

        # TODO 4：如果已经到达文本末尾，结束循环
        if end == len(text):
           break
        # TODO 5：计算下一个 start
        start = start + chunk_size - overlap

    return chunks


#10-4：将文本块与 PDF 页码绑定

def build_chunks(pages):
    all_chunks = []
    chunk_id = 0

    for page_item in pages:
        page_num = page_item["page"]
        text = page_item["text"]

        # TODO 1：调用 split_text()，获得 chunk_list
        chunk_list = split_text(text)

        # TODO 2：遍历 chunk_list
        for chunk_text in chunk_list:

            # TODO 3：创建包含以下三个字段的字典
            # "chunk_id"、"page"、"text"
            chunk_dict = {
                "chunk_id":chunk_id,
                "page":page_num,
                "text":chunk_text
            }

            # TODO 4：将字典添加到 all_chunks
            all_chunks.append(chunk_dict)

            # TODO 5：chunk_id 加 1
            chunk_id += 1
    # TODO 6：返回 all_chunks
    return all_chunks

# # TODO 6：打印前两个 Chunk 的页码及文本前100个字符
# for chunk_dict in all_chunks[:2]:
#     print(f"Chunk {chunk_dict['chunk_id']}，PDF第{chunk_dict['page']}页")
#     print(chunk_dict["text"][:100])

#10-5：实现论文检索功能
def search_chunks(all_chunks, keyword):
    results = []

    keyword = keyword.strip()

    # TODO 1：判断 keyword 是否为空
    if not keyword:
        return results

    # TODO 2：遍历 all_chunks
    for chunk in all_chunks:

        # TODO 3：不区分大小写地检查
        # keyword 是否出现在 chunk["text"] 中
        if keyword.casefold() in chunk["text"].casefold():

            # TODO 4：把匹配的 chunk 加入 results
            results.append(chunk)

    return results

# 只有直接运行本文件时，才执行以下代码
if __name__ == "__main__":
    pages = extract_pdf_pages("data/sample.pdf")
    all_chunks = build_chunks(pages)

    print("PDF 总页数：", len(pages))
    print("Chunk 总数：", len(all_chunks))
    print("首个 Chunk ID：", all_chunks[0]["chunk_id"])
    print("最后 Chunk ID：", all_chunks[-1]["chunk_id"])
