class Paper:

    def __init__(self, title, year, author,journal):
        self.title = title
        self.year = year
        self.author = author
        self.journal = journal

    def show_info(self):
        print("论文标题：", self.title)
        print("年份：", self.year)
        print("作者：", self.author)
        print("期刊：",self.journal)

    def is_recent(self):
        if int(self.year) >= 2024:
            print("这是一篇较新的论文")
        else:
            print("这是一篇较早的论文")

    def to_dict(self):
        return {
            "title": self.title,
            "year": self.year,
            "author": self.author,
            "journal": self.journal
        }


paper1 = Paper(
    "Wearable Sensor",
    "2026",
    "Tom",
    "Advanced Materials"
)

paper2 = Paper(
    "AI Agent",
    "2025",
    "Jack",
    "Advanced Materials"
)


paper1.show_info()
paper1.is_recent()

print("----------")

paper2.show_info()
paper2.is_recent()

print("----------")

print(paper1.to_dict())