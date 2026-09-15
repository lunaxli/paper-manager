import requests
import json

def get_post(post_id):
    url = f"https://jsonplaceholder.typicode.com/posts/{post_id}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data

    except requests.RequestException as error:
        print("请求失败: ",error)
        return None

post = get_post(1)

if post is not None:
    print("标题：",post["title"])

    with open("post.json","w",encoding="utf-8") as file:
        json.dump(
            post,
            file,
            ensure_ascii=False,
            indent=4
        )
    print("保存成功！")
