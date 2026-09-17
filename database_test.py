import sqlite3


connection = sqlite3.connect("papers.db")
cursor = connection.cursor()


cursor.execute("""
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    year INTEGER
)
""")


while True:

    print("\n=== 数据库论文管理 ===")
    print("1. 添加论文")
    print("2. 查看论文")
    print("3. 搜索论文")
    print("4. 删除论文")
    print("5. 修改论文年份")
    print("6. 退出")

    choice = input("请选择：")


    if choice == "1":

        title = input("请输入论文标题：")
        year = int(input("请输入论文年份："))

        cursor.execute(
            "INSERT INTO papers (title, year) VALUES (?, ?)",
            (title, year)
        )

        connection.commit()

        print("添加成功！")


    elif choice == "2":

        cursor.execute("SELECT * FROM papers")

        papers = cursor.fetchall()

        if not papers:
            print("当前还没有论文")
        else:
            for paper in papers:
                print(
                    paper[0],
                    paper[1],
                    paper[2]
                )


    elif choice == "3":

        keyword = input("请输入搜索关键词：")

        cursor.execute(
            "SELECT * FROM papers WHERE title LIKE ?",
            (f"%{keyword}%",)
        )

        papers = cursor.fetchall()

        if not papers:
            print("没有找到相关论文")
        else:
            for paper in papers:
                print(
                    paper[0],
                    paper[1],
                    paper[2]
                )


    elif choice == "4":

        paper_id = int(
            input("请输入要删除的论文ID：")
        )

        cursor.execute(
            "DELETE FROM papers WHERE id = ?",
            (paper_id,)
        )

        connection.commit()

        if cursor.rowcount > 0:
            print("删除成功！")
        else:
            print("没有找到这个ID的论文")


    elif choice == "5":

        paper_id = int(input("请输入论文ID："))
        new_year = int(input("请输入新年份："))

        cursor.execute(
            "UPDATE papers SET year = ? WHERE id = ?",
            (new_year, paper_id)
        )

        connection.commit()

        if cursor.rowcount > 0:
            print("修改成功！")
        else:
            print("没有找到这个ID的论文")


    elif choice == "6":

        print("程序结束")
        break


    else:
        print("输入错误")


connection.close()