import json

def load_papers():
    try:
        with open("papers.json", "r", encoding="utf-8") as file:
            papers = json.load(file)
            return papers

    except FileNotFoundError:
        return []


def save_papers(papers):
    with open("papers.json", "w", encoding="utf-8") as file:
        json.dump(
            papers,
            file,
            ensure_ascii=False,
            indent=4
        )