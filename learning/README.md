# Python 学习练习

这里保存独立的早期练习，业务代码和 tests/ 均不引用它们。文件名保留学习历史；这些脚本不是 pytest 自动化测试。

| 文件 | 内容 | 运行时影响 |
|---|---|---|
| api_test.py | requests、公开 HTTP API 与 JSON 写入 | 请求 JSONPlaceholder，覆盖根目录 post.json |
| class_test.py | 类、实例方法与字典转换 | 打印内置示例 |
| database_test.py | SQLite 交互式 CRUD | 创建或修改根目录 papers.db |
| sql_test.py | SQL INSERT | 向根目录 papers.db 插入记录，需先存在 papers 表 |
| embedding_test.py | 向量、余弦相似度与 Top-K | 可能下载模型并占用缓存 |
| llm_test.py | DeepSeek SDK 调用 | 读取根目录 .env，实际请求 API 并消耗额度 |

在项目根目录激活原虚拟环境后按需执行，例如：

```powershell
python learning/class_test.py
python learning/database_test.py
```

也支持 `python -m learning.class_test`。直接进入 learning/ 后运行脚本时，数据库、JSON 和 .env 仍定位到项目根目录，避免意外新建另一份数据。SQL 练习保留原有数据库行为，运行前请自行备份本地数据库。

旧命令如 `python class_test.py` 请改为 `python learning/class_test.py`。未在根目录保留同名转发文件，以免再次被 pytest 误收集。

`main.py`、`paper_service.py`、`storage.py` 组成仍可独立使用的 JSON 命令行程序，继续放在根目录。当前 FastAPI/Agent/RAG 业务模块也保留在根目录。

从项目根目录执行 `python -m pytest` 会收集 tests/ 中的全部现有自动化测试；learning/ 已排除。不要将练习脚本作为测试文件显式传给 pytest。
