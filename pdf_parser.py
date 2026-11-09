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


def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap需要满足0 <= overlap < chunk_size")

    while start < len(text):
        end = min(start+chunk_size,len(text))

        chunk = text[start:end]

        chunks.append(chunk)

        if end == len(text):
           break
        start = start + chunk_size - overlap

    return chunks


def build_chunks(pages):
    all_chunks = []
    chunk_id = 0

    for page_item in pages:
        page_num = page_item["page"]
        text = page_item["text"]

        chunk_list = split_text(text)

        for chunk_text in chunk_list:

            chunk_dict = {
                "chunk_id":chunk_id,
                "page":page_num,
                "text":chunk_text
            }

            all_chunks.append(chunk_dict)

            chunk_id += 1
    return all_chunks
