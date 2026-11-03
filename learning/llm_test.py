import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 1. 读取 .env 中的环境变量
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没有找到 DEEPSEEK_API_KEY")

# 2. 创建 API 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# 3. 向模型发送请求
response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "user",
            "content": "你好，请用一句话介绍什么是 RAG。"
        }
    ],
    extra_body={"thinking": {"type": "disabled"}},
    max_tokens=100
)

# 4. 打印模型回答
print(response.choices[0].message.content)

print("输入 Token：", response.usage.prompt_tokens)
print("输出 Token：", response.usage.completion_tokens)
print("总 Token：", response.usage.total_tokens)
