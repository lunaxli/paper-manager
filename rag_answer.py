import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def generate_answer(question, context):
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("缺少 DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    system_prompt = """
    你是一名科研论文分析助手。

    回答规则：
    1. 只能依据用户提供的论文片段回答。
    2. 不要编造论文中不存在的方法、数据或结论。
    3. 回答时标注支持结论的 PDF 页码。
    4. 如果检索片段没有足够证据，请明确说明无法确定。
    5. 使用中文回答，保留必要的英文专业术语。
    """


    user_prompt = f"""
    以下是从论文中检索到的相关片段：

    {context}

    请根据以上论文片段回答这个问题：

    {question}
    """

    response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "system",
            "content":system_prompt
        },
        {
            "role":"user",
            "content":user_prompt
        }
    ],
    extra_body={"thinking": {"type": "disabled"}},
    max_tokens=800
    )

    # 提取模型回答
    answer = response.choices[0].message.content
    # 打印模型回答
    return answer