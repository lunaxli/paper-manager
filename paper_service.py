from storage import save_papers


def add_paper(papers):
    title = input("请输入论文标题：")
    year = input("请输入论文年份：")

    paper = {
        "title": title,
        "year": year
    }

    papers.append(paper)

    save_papers(papers)

    print("添加成功！")

def show_papers(papers):
    if len(papers) == 0:
        print("当前还没有论文")
        return

    print("\n当前论文：")

    for paper in papers:
        print("论文名称：", paper["title"])
        print("年份：", paper["year"])

def search_paper(papers):
    keyword = input("请输入搜索关键词：")

    found = False

    for paper in papers:
        if keyword.lower() in paper["title"].lower():
            found = True

            print("找到论文：")
            print("论文名称：", paper["title"])
            print("年份：", paper["year"])

    if not found:
        print("没有找到相关论文")

def delete_paper(papers):
    title = input("请输入要删除的论文标题：")

    found = False

    for paper in papers:
        if title.lower() == paper["title"].lower():

            papers.remove(paper)

            save_papers(papers)

            found = True

            print("删除成功！")

            break

    if not found:
        print("没有找到这篇论文")

