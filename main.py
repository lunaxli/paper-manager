from storage import load_papers

from paper_service import (
    add_paper,
    show_papers,
    search_paper,
    delete_paper
)


papers = load_papers()


print("=== 论文管理系统 ===")


while True:

    print("\n请选择操作：")
    print("1. 添加论文")
    print("2. 查看论文")
    print("3. 退出")
    print("4. 搜索论文")
    print("5. 删除论文")

    choice = input("请输入选项：")

    if choice == "1":
        add_paper(papers)

    elif choice == "2":
        show_papers(papers)

    elif choice == "3":
        print("程序结束")
        break

    elif choice == "4":
        search_paper(papers)

    elif choice == "5":
        delete_paper(papers)

    else:
        print("输入错误，请重新输入")