import sqlite3


connection = sqlite3.connect("papers.db")

cursor = connection.cursor()


title = input("请输入论文标题：")

year = int(input("请输入论文年份："))


cursor.execute(
    "INSERT INTO papers (title, year) VALUES (?, ?)",
    (title, year)
)


connection.commit()

print("添加成功！")


connection.close()