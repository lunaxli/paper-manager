# ResearchAgent

基于 FastAPI、RAG 和 LLM Tool Calling 构建的科研论文智能问答系统。

## 项目简介

ResearchAgent 是一个面向科研论文阅读场景的 AI 应用，支持从 PDF 论文中检索相关内容，并结合大语言模型生成回答。

项目实现了两种问答模式：

- **RAG 问答**：先检索论文片段，再使用大模型生成回答。
- **Agent 问答**：由大模型自主判断是否调用论文检索工具，并根据工具返回的信息生成最终回答。

系统可以返回论文页码、文本片段 ID 和向量相似度，帮助用户定位相关论文内容。

## 技术栈

| 类别 | 技术 |
|---|---|
| 后端框架 | FastAPI、Pydantic |
| 大语言模型 | DeepSeek（OpenAI 兼容接口） |
| Agent | Function Calling / Tool Calling |
| RAG | FastEmbed、NumPy |
| PDF 解析 | PyMuPDF |
| 数据库 | SQLite |
| 自动化测试 | pytest、FastAPI TestClient、Mock |

## 核心功能

1. PDF 文本提取与分块
2. Embedding 向量生成与相似度检索
3. 基于论文内容的 RAG 问答
4. 大模型自主决策的工具调用
5. 多次工具调用及结果整合
6. 返回检索来源信息
7. FastAPI REST API
8. 自动化测试与异常处理

## Agent 工作流程

用户问题 → DeepSeek 判断是否需要工具 → 调用 search_paper → 检索论文片段 → 返回工具结果 → DeepSeek 生成最终回答。

如果大模型没有发起工具调用，则直接返回模型回答。

当前实现以单轮工具执行为主，不包含多轮自主规划或多 Agent 协作。

## 项目结构

```text
paper-manager/
├── api.py
├── agent_runner.py
├── agent_tools.py
├── pdf_parser.py
├── rag_retrieval.py
├── rag_answer.py
├── main.py
├── paper_service.py
├── storage.py
├── data/
├── tests/
├── learning/          # 独立的 Python 学习练习
├── pytest.ini         # 仅收集自动化测试
├── requirements.txt
├── .gitignore
└── README.md
```

## 本地运行

**1. 创建并激活虚拟环境**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**2. 安装依赖**

```powershell
python -m pip install -r requirements.txt
```

**3. 配置 DeepSeek API Key**

在 PowerShell 当前会话中设置环境变量：

```powershell
$env:DEEPSEEK_API_KEY="你的API_KEY"
```

不要把真实 API Key 提交到 GitHub。

**4. 准备论文**

将用于测试的 PDF 论文放到：

```text
data/sample.pdf
```

当前版本使用本地固定 PDF 文件作为知识来源。论文 PDF 不随公开仓库分发。

**5. 启动服务**

```powershell
python -m uvicorn api:app --reload
```

打开 Swagger API 文档：

http://127.0.0.1:8000/docs

## API 示例

### RAG 问答

`POST /rag/ask`

```json
{
  "question": "论文中葡萄糖的检测限是多少？"
}
```

### Agent 问答

`POST /agent/ask`

```json
{
  "question": "根据论文实验数据，葡萄糖的检测限是多少？"
}
```

返回内容包含 `answer` 和 `sources`，其中 `sources` 表示检索命中的候选论文片段，并非逐句验证后的引用。

## 运行测试

```powershell
python -m pytest -v
```

测试覆盖 API 参数校验、RAG 流程、Agent Tool Calling、模型异常处理等功能。

## 当前限制

- 当前版本仅使用本地固定论文文件，尚不支持多用户上传和管理独立知识库。
- Agent 使用自定义 Tool Calling 流程，尚未集成 LangGraph。
- 检索来源是候选证据，尚未实现回答与证据的逐句一致性验证。
- 当前主要面向本地运行与开发测试。

## 后续规划

- 使用 LangGraph 实现状态化 Agent 工作流
- 使用 MCP 标准封装论文检索工具
- 完善数据库持久化及部署能力
- 增加 Agent 执行日志与效果评估

## 学习练习与兼容入口

独立的 HTTP、类、SQLite、Embedding 和 LLM 练习已归入 `learning/`，详见 [学习练习说明](learning/README.md)。这些脚本具有交互、写数据或联网行为，不作为 pytest 测试收集。

旧 JSON 命令行入口继续使用 `python main.py`；API 服务继续使用 `python -m uvicorn api:app --reload`。请在项目根目录运行这两个入口，以沿用原有的相对数据路径。
