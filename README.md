# ResearchAgent

**基于 FastAPI、RAG、DeepSeek、LangGraph 和 MCP 的科研论文智能问答系统**

ResearchAgent 是一个面向科研论文阅读场景的 AI Agent 项目，利用 PDF 文本解析、向量语义检索、大语言模型工具调用和 MCP 协议，实现基于论文内容的智能问答与证据检索。

项目采用渐进式架构，从固定 RAG 演进到自定义 Tool Calling Agent、LangGraph 状态化 Agent，最终实现 **LangGraph + MCP 的独立工具服务调用**。

## 1. 项目简介

科研论文通常包含大量专业术语、实验数据和研究结论。直接使用通用大语言模型回答论文相关问题，可能缺乏准确的文献依据。

ResearchAgent 通过构建本地论文检索工具，让大语言模型能够根据用户问题检索论文片段，并结合检索结果生成回答。

项目提供四种问答模式：

| 模式 | API | 技术实现 |
|---|---|---|
| 固定 RAG | `POST /rag/ask` | 先检索论文，再生成回答 |
| V1 Agent | `POST /agent/ask` | DeepSeek Tool Calling |
| V2 Agent | `POST /graph/ask` | LangGraph 状态化工作流 |
| V3 Agent | `POST /graph-mcp/ask` | LangGraph + MCP Client/Server |

### 核心特性

- 基于 PyMuPDF 的 PDF 文本提取与分块
- 基于 FastEmbed 和 NumPy 的向量语义检索
- DeepSeek Function Calling / Tool Calling
- LangGraph 状态管理与条件路由
- MCP Server 工具注册与独立进程通信
- LangGraph 通过 MCP Client 调用论文检索工具
- 多工具调用与检索来源去重排序
- FastAPI REST API 与 Pydantic 数据校验
- 异常处理、日志记录和环境变量配置
- pytest、Mock 和 FastAPI TestClient 自动化测试

## 2. 技术栈

| 类别 | 技术 |
|---|---|
| 开发语言 | Python |
| 后端框架 | FastAPI、Pydantic、Uvicorn |
| 大语言模型 | DeepSeek（OpenAI 兼容 API） |
| Agent 工作流 | LangGraph |
| Agent 工具调用 | Tool Calling |
| 工具通信协议 | Model Context Protocol（MCP） |
| MCP SDK | MCP Python SDK 2.3.0 |
| Embedding | FastEmbed |
| 向量计算 | NumPy |
| PDF 解析 | PyMuPDF |
| 自动化测试 | pytest、httpx、unittest.mock |
| 配置与日志 | 环境变量、Python logging |

## 3. 系统架构

ResearchAgent 保留四种问答模式，用于展示 AI 应用架构的逐步演进。

### 3.1 固定 RAG

```text
用户问题
   |
   v
Embedding 向量化
   |
   v
论文片段语义检索
   |
   v
构建检索上下文
   |
   v
DeepSeek 生成回答
```

固定 RAG 每次都会执行论文检索，适合明确需要参考本地文档回答的问题。

### 3.2 V1：Tool Calling Agent

```text
用户问题
   |
   v
DeepSeek 判断是否调用工具
   |
   +-- 无需工具 --> 直接回答
   |
   +-- 需要工具 --> search_paper()
                       |
                       v
                   检索论文
                       |
                       v
                  DeepSeek 生成回答
```

V1 使用自定义 Python 代码完成工具决策、参数解析、工具执行及最终回答生成。

### 3.3 V2：LangGraph Agent

V2 引入 LangGraph，将 Agent 执行过程组织为显式状态图。

```text
START
   |
   v
prepare_question
   |
   v
decide_tool
   |
   +-- direct --------------------> END
   |
   +-- search
         |
         v
     execute_tools
         |
         v
     generate_final_answer
         |
         v
        END
```

主要节点：

- `prepare_question`：处理和校验用户问题
- `decide_tool`：让 DeepSeek 决定是否执行检索
- `execute_tools`：执行本地论文检索工具
- `generate_final_answer`：根据工具结果生成回答

V2 支持多个工具调用，并对来源按照 `(page, chunk_id)` 去重，保留较高相似度分数。

### 3.4 V3：LangGraph + MCP

V3 在 V2 的基础上，引入 MCP 工具通信机制。

```text
用户问题
   |
   v
FastAPI /graph-mcp/ask
   |
   v
LangGraph Agent
   |
   v
DeepSeek 工具决策
   |
   +-- 不需要检索 --> 直接回答
   |
   +-- 需要检索
          |
          v
     LangGraph MCP Tool Node
          |
          v
        MCP Client
          |
          | stdio / MCP
          v
        MCP Server
          |
          v
      search_paper()
          |
          v
      PDF 向量检索
          |
          v
     context + sources
          |
          v
    DeepSeek 生成最终回答
          |
          v
         END
```

MCP Server 将现有 `search_paper()` 封装为标准 MCP Tool。

LangGraph 不再直接执行底层检索函数，而是通过 MCP Client 向独立 MCP Server 发起工具请求。

当前 V3 使用本地 stdio 通信方式，MCP Server 以独立子进程运行。

**设计目标：** 将 Agent 工作流与工具服务的通信方式解耦，使论文检索工具可以通过标准协议被调用。

## 4. 核心实现

### 4.1 PDF 解析与 RAG 检索

使用 PyMuPDF 提取论文文本，并保留对应页码。

使用 FastEmbed 的 `BAAI/bge-small-en-v1.5` 模型生成文本 Embedding，通过 NumPy 计算语义相似度，检索相关文本片段。

检索结果包含：

- `page`：论文页码
- `chunk_id`：文本片段编号
- `score`：语义相似度分数
- `preview`：检索片段预览

### 4.2 MCP 工具服务

`mcp_server.py` 负责注册并提供论文检索工具：

```python
@mcp.tool(name="search_paper")
def search_paper_mcp(query: str) -> dict[str, Any]:
    ...
```

MCP Server 复用现有论文检索函数，不重新实现 PDF 解析和向量检索算法。

使用 `lru_cache` 缓存检索资源，避免同一服务器进程内重复初始化。

### 4.3 MCP Client

`mcp_client.py` 使用 MCP Python SDK，通过 stdio 与独立 MCP Server 通信。

实现：

- MCP Server 子进程启动
- MCP 工具发现
- `search_paper` 工具调用
- 工具调用失败处理
- 结构化结果校验
- 异步资源管理

### 4.4 LangGraph MCP 工作流

`graph_agent.py` 保留 V2 原有节点，并新增 MCP 工具节点和 V3 工作流创建函数。

V3 使用 `graph.ainvoke()` 执行异步工作流，在 MCP Client 连接有效期间完成工具调用。

### 4.5 FastAPI 接口

`api.py` 对外提供 RAG、V1、V2 和 V3 四种问答接口。

V3 支持：

- 输入参数校验
- DeepSeek API Key 检查
- MCP Client 生命周期管理
- LangGraph 异步执行
- DeepSeek 和 MCP 异常处理
- 回答有效性检查
- 结构化检索来源返回

## 5. 项目结构

```text
paper-manager/
├── api.py                 # FastAPI API 入口
├── config.py              # DeepSeek 和日志配置
├── graph_agent.py         # V2/V3 LangGraph 工作流
├── agent_runner.py        # V1 Tool Calling
├── agent_tools.py         # 论文检索工具
├── mcp_server.py          # V3 MCP Server
├── mcp_client.py          # V3 MCP Client
├── pdf_parser.py          # PDF 文本解析
├── rag_retrieval.py       # 向量检索
├── rag_answer.py          # RAG 回答生成
├── data/                  # 本地 PDF 数据
├── tests/                 # 自动化测试
├── requirements.txt       # Python 依赖
├── pytest.ini             # pytest 配置
├── .gitignore
└── README.md
```

## 6. 本地运行

### 6.1 环境准备

建议使用 Python 3.11 或更高版本。

创建并激活虚拟环境（Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

### 6.2 配置 DeepSeek API Key

在 PowerShell 中执行：

```powershell
$env:DEEPSEEK_API_KEY="你的API_KEY"
```

可选配置：

```powershell
$env:DEEPSEEK_MODEL="deepseek-flash"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MAX_TOKENS="700"
$env:LOG_LEVEL="INFO"
```

模型名称应根据实际可用的 DeepSeek 服务配置。

请勿将真实 API Key 上传到公开仓库。

### 6.3 准备论文

将用于检索的 PDF 放入：

```text
data/sample.pdf
```

当前项目使用本地固定论文文件，原始论文不随公开仓库分发。

### 6.4 启动 FastAPI

在项目根目录执行：

```powershell
python -m uvicorn api:app --reload
```

访问 Swagger UI：

http://127.0.0.1:8000/docs

### 6.5 独立测试 MCP Client

运行：

```powershell
python mcp_client.py
```

客户端会自动启动 MCP Server 子进程，并通过 MCP 协议发现和调用 `search_paper` 工具。

**无需额外手动启动 MCP Server。**

此演示使用本地论文向量检索，不需要调用 DeepSeek。

## 7. API 示例

### 固定 RAG

`POST /rag/ask`

```json
{
  "question": "论文中葡萄糖的检测限是多少？"
}
```

### V1 Tool Calling Agent

`POST /agent/ask`

```json
{
  "question": "根据论文实验数据，葡萄糖的检测限是多少？"
}
```

### V2 LangGraph Agent

`POST /graph/ask`

```json
{
  "question": "根据论文，葡萄糖检测的检测限是多少？请提供依据。"
}
```

### V3 LangGraph + MCP Agent

`POST /graph-mcp/ask`

```json
{
  "question": "根据论文，葡萄糖检测的检测限是多少？请提供依据。"
}
```

使用项目测试论文时，真实调用曾成功返回以下结果：

**回答：** 葡萄糖检测限为 **0.4631 μM**。

对应候选证据：

- 论文第 6 页
- `chunk_id=66`
- 相关实验数据和检测限描述

API 响应结构：

```json
{
  "answer": "根据论文，葡萄糖检测限为 0.4631 μM。",
  "sources": [
    {
      "page": 6,
      "chunk_id": 66,
      "score": 0.79,
      "preview": "相关论文片段预览"
    }
  ]
}
```

示例仅展示响应格式。实际回答、来源数量、相似度和文本内容会随查询及论文内容变化。

**注意：** `sources` 是语义检索命中的候选证据，并不代表已经完成回答与证据的逐句事实验证。

## 8. 自动化测试

项目使用 pytest、FastAPI TestClient 和 Mock 验证核心逻辑。

测试覆盖：

- API 请求校验
- 固定 RAG
- Tool Calling 和工具执行
- LangGraph 条件路由
- MCP Server 工具注册
- MCP Client 发现与调用工具
- LangGraph + MCP 完整工作流
- DeepSeek API 异常处理
- MCP 通信异常处理
- 结构化响应校验
- 配置与日志行为

运行全部测试：

```powershell
python -m pytest tests/ -q
```

检查 Python 依赖：

```powershell
python -m pip check
```

核心测试使用 Mock 模拟 DeepSeek 和 MCP 通信，减少外部服务依赖。

项目同时进行了真实的 MCP 工具调用以及 `/graph-mcp/ask` 端到端验证。

## 9. V1、V2、V3 技术演进

| 对比 | V1 | V2 | V3 |
|---|---|---|---|
| 工具决策 | DeepSeek | DeepSeek | DeepSeek |
| 工作流管理 | Python 代码 | LangGraph | LangGraph |
| 工具执行 | 本地函数 | 本地函数 | MCP Server |
| 工具通信 | 函数调用 | 函数调用 | MCP stdio |
| 状态管理 | 手动管理 | StateGraph | StateGraph |
| 异步 MCP 调用 | 无 | 无 | 支持 |

### 为什么引入 LangGraph？

将 Agent 的决策、工具执行和最终回答拆分为独立节点，使状态传递、条件路由和工作流测试更加明确。

### 为什么引入 MCP？

将工具调用从应用内部函数执行扩展为标准化的工具服务通信。

通过 MCP Server 提供论文检索能力，LangGraph 可以通过 MCP Client 调用该工具，而不必直接调用底层检索函数。

### 为什么保留三个版本的 Agent？

V1、V2、V3 展示同一 AI 应用从自定义工具调用到状态化工作流，再到标准化工具服务集成的技术演进。

这种设计便于对比架构复杂度、工具耦合程度和扩展方式。

## 10. 当前限制

- 当前主要支持本地固定 PDF 文件。
- 尚不支持多用户上传和管理独立知识库。
- 当前使用本地向量检索，尚未引入专用向量数据库。
- V3 采用本地 stdio MCP Server，每个 API 请求建立独立 MCP 会话，存在进程启动和检索资源初始化开销。
- 当前主要支持单轮 Agent 决策及工具执行，尚未实现复杂多轮自主规划。
- 检索来源为候选证据，未实现逐句引用验证。
- 尚未建立系统化的 RAG 检索质量与 Agent 回答评估体系。
- 当前以本地运行和学习演示为主，尚未完成生产环境部署。

## 11. 后续方向

后续优化将以真实使用需求和项目展示价值为导向：

- 增加 RAG 检索与回答效果评估
- 扩展多 PDF 文档处理能力
- 完善 Agent 演示案例和项目部署
- 根据实际需求接入更多 MCP 工具

暂不进行不必要的数据库重构或复杂多 Agent 开发。

---

**项目定位：** 面向 Python 后端、AI 应用开发和 Agent 开发方向的工程实践项目，重点展示 RAG、Tool Calling、LangGraph、MCP、FastAPI 和自动化测试的完整技术链路。