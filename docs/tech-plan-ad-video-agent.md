# 广告设计 / 视频剪辑 AI Agent 技术方案

> 目标读者：计算机图形学背景、无 agent 后端经验的开发者
> 技术栈：Deep Agents（harness）+ LangGraph（runtime）+ LangChain（framework）+ LangSmith（观测）
> 日期：2026-07-02

---

## 1. 需求梳理

公司业务：广告设计、视频剪辑。原始需求模糊，翻译成 agent 的输入/输出如下：

| 输入 | 输出 |
|------|------|
| 文案稿 / 分镜脚本 | 结构化时间线（镜头列表：时刻、旁白、画面描述） |
| 时间线中的某个镜头 | 文生图/文生视频提示词（prompt） |
| 提示词 | 插画（调用生图 API） |
| 静态插画 + 动效意图 | 动效方案：动效描述 → Lottie/CSS/Remotion 代码 → 图生视频 |
| 已有视频成片 | 时刻定位、内容理解（转写 + 抽帧 + 视觉模型） |
| 公司项目文件、品牌规范、历史案例 | 检索问答、结合上下文的推理与建议 |

隐含需求（客户没说但必须有）：
- **人审**：设计产出主观性强，生成前确认 prompt、生成后人工挑选；
- **产物管理**：每个项目的 prompt、图、动效文件要有目录规范，可回溯；
- **项目连续性**：跨会话记住"这个客户喜欢什么风格"；
- **非技术人员可维护**：品牌规范、prompt 模板必须是普通人能编辑的格式（Markdown）。

## 2. 可行性分析

| 功能 | 可行性 | 说明 | 建议阶段 |
|------|--------|------|----------|
| 脚本 → 结构化时间线 | ★★★★★ | LLM 结构化输出，成熟 | MVP |
| 分镜 → 生图提示词 | ★★★★★ | 纯 LLM + 模板（skill），你自己就能调优 | MVP |
| 插画生成 | ★★★★☆ | 调 fal.ai / Replicate / ComfyUI API，工程量小 | MVP |
| 公司资料 RAG 问答 | ★★★★☆ | LangChain 标准 RAG 管线 | 阶段 2 |
| 动效：建议文档 / 代码生成 | ★★★★☆ | 生成 Lottie JSON、CSS/SVG、Remotion(React) 组件代码——对 CG 背景非常顺手 | 阶段 2 |
| 动效：图生视频 | ★★★☆☆ | 调可灵/Runway/fal 上的 I2V 模型，能用但成本高、可控性一般 | 阶段 3 |
| 视频内容理解（时刻定位） | ★★★☆☆ | whisper 转写拿时间轴 + ffmpeg 抽帧 + 视觉模型看帧；纯视觉理解贵且不稳，用"字幕时间轴对齐"兜底 | 阶段 3 |
| 全自动操控 PR/AE 成片 | ★★☆☆☆ | 脆、难维护。**不建议做**；改为输出剪辑指令清单 / EDL / FCPXML 给剪辑师导入 | 不做/远期 |

结论：**整体可行**。MVP（脚本拆解 + 提示词 + 生图 + 人审 + 产物落盘）用 Deep Agents 现成能力 + 2~3 个自定义工具即可完成，预计 1~2 周出可演示版本。

## 3. 总体架构

```
┌─────────────────────────────────────────────────────────┐
│ 交互层：CLI（MVP）→ agent-chat-ui / LangGraph dev（阶段3）│
├─────────────────────────────────────────────────────────┤
│ 编排层：Deep Agent 主代理（"制片/导演"）                  │
│   内置：write_todos 规划 / 文件工具 / task 子代理委派      │
│   配置：skills 目录、interrupt_on 人审、checkpointer、store│
├──────────────┬──────────────┬──────────────┬────────────┤
│ 子代理        │              │              │            │
│ script-      │ prompt-      │ illustrator  │ motion-    │
│ analyst      │ artist       │ (生图)       │ designer   │
│ 脚本→时间线   │ 镜头→提示词  │              │ 动效       │
│ (LangGraph图 │ (加载prompt  │              │            │
│  结构化输出)  │  模板skill)  │              │            │
│         ┌────┴─────┐                                    │
│         │ knowledge │ ← RAG 检索公司资料                  │
├─────────┴──────────┴────────────────────────────────────┤
│ 工具层：自定义 @tool（ffmpeg / whisper / 生图API / I2V）  │
│         + MCP（langchain-mcp-adapters）：搜索、fal 等     │
├─────────────────────────────────────────────────────────┤
│ 数据层：项目文件目录（FilesystemBackend）                 │
│         向量库 Chroma（RAG）│ SQLite checkpointer │ Store │
├─────────────────────────────────────────────────────────┤
│ 观测层：LangSmith（全链路 trace、评估、调试）              │
└─────────────────────────────────────────────────────────┘
```

## 4. 关键设计决策

### 4.1 为什么用 Deep Agents 而不是裸写 LangGraph

需求命中 Deep Agents 内置能力的全部四项：多步规划（TodoList）、跨会话文件管理（Filesystem 工具）、子代理委派（SubAgent）、持久记忆（Store）。裸写 LangGraph 意味着这些全部自己实现——对无 agent 后端经验的开发者是数周额外工作量。

**分工原则**：主流程交给 harness；只有需要确定性控制流的局部（如"脚本解析→校验→重试"这类固定管线）下沉为 LangGraph `StateGraph`，编译后注册为命名子代理挂进 Deep Agents。

### 4.2 产物即文件

所有产出（时间线 JSON、prompt.md、插画 PNG、动效代码）通过内置文件工具写入项目目录，用 `FilesystemBackend(root_dir=项目根)`。好处：设计师直接在 Finder/网盘里看结果，不需要懂任何技术；agent 重启后上下文可从文件恢复。

建议目录规范：

```
projects/<客户>/<项目>/
  brief/            # 输入：文案稿、参考图
  timeline.json     # 结构化时间线
  shots/
    010/
      prompt.md     # 提示词（含负向词、参数）
      draft-*.png   # 候选图
      final.png     # 人工选定
      motion.md     # 动效方案 / motion.json (Lottie) / Motion.tsx (Remotion)
  notes.md          # 客户反馈、风格结论
```

### 4.3 公司知识的两条通道

| 通道 | 放什么 | 维护者 |
|------|--------|--------|
| **Skills**（`skills/` 下的 SKILL.md） | 品牌规范、各生图模型的 prompt 模板、动效规范、交付 checklist | 公司运营/设计人员（纯 Markdown，写完即生效） |
| **RAG**（向量库） | 历史项目文档、大量案例、合同/需求文本 | 定期批量入库脚本 |

Skills 是按需加载（progressive disclosure），不占常驻上下文；这是让"没有技术人员的公司"能自己养护 agent 的关键机制。

示例 skill：

```markdown
---
name: midjourney-prompt-style
description: 本公司广告插画的 Midjourney 提示词规范与风格词库
---
# 提示词规范
## 固定后缀
--ar 16:9 --style raw ...
## 品牌色
主色 #FF6A00，禁用纯黑背景...
```

### 4.4 人审（Human-in-the-loop）

所有**花钱**或**对外**的操作必须过人审：

```python
interrupt_on={
    "generate_image": True,      # 生图前确认 prompt 和张数
    "generate_video": True,      # I2V 更贵，必须确认
}
```

interrupt 需要 checkpointer（见 4.6）。生成后的"选图"环节也走 interrupt：agent 出 3 张候选 → 暂停 → 人选 → `Command(resume=...)` 继续。

### 4.5 MCP vs 自定义 tool 的取舍

原则：**核心生成链路用自定义 `@tool` 封装**（可控、可加参数校验和成本上限），**外围能力用 MCP**（换现成的生态）。

| 能力 | 接入方式 | 说明 |
|------|----------|------|
| 文生图（fal.ai / Replicate） | 自定义 @tool（MVP）；fal 官方 MCP（后续可换） | 核心链路，自己封装 20 行代码，可控 |
| ComfyUI 本地工作流 | 自定义 @tool 调 ComfyUI HTTP API | CG 背景可自建工作流，质量上限最高 |
| ffmpeg（抽帧/裁切/拼接） | 自定义 @tool（subprocess 封装） | 社区 ffmpeg MCP 质量参差，自封装更稳 |
| 语音转写（whisper） | 自定义 @tool | 拿到带时间戳的字幕，是"时刻对齐"的地基 |
| 网络搜索（竞品/素材调研） | Tavily MCP 或 langchain tavily 工具 | 现成 |
| 文件读写 | **不需要接** —— Deep Agents 内置 | |

MCP 接入统一走 `langchain-mcp-adapters`：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
client = MultiServerMCPClient({...})
tools = await client.get_tools()
agent = create_deep_agent(model=..., tools=[*custom_tools, *tools], ...)
```

### 4.6 持久化与记忆

- `checkpointer`：MVP 用 `MemorySaver`，阶段 2 换 SQLite（`langgraph-checkpoint-sqlite`），部署换 Postgres。作用：会话可断点续跑、interrupt 依赖它。
- `store`：跨会话记忆（"客户 A 偏好扁平插画风"）。MVP 用 `InMemoryStore`，后换持久实现。
- `thread_id` 约定：`<客户>-<项目>`，同一项目的多次会话共享上下文。

### 4.7 模型选择

- 主代理 / 子代理：`claude-sonnet-4-6`（工具调用与长任务表现好）；
- 看图/看帧：同用 Claude 的视觉能力，省一套集成；
- 成本敏感的批量小任务（如逐镜头 prompt 初稿）：可经 OpenRouter 换便宜模型（仓库已装 `langchain-openrouter`）。

### 4.8 动效的三级降级策略

1. **动效方案文档**（MVP 附带）：每个镜头输出"入场/强调/出场 + 缓动 + 时长"建议，剪辑师照做；
2. **代码生成**（阶段 2，最适合你的背景）：Lottie JSON、CSS/SVG 动画、**Remotion**（React 编程式视频，agent 写组件代码 → `npx remotion render` 出片，可精确到帧）；
3. **图生视频 API**（阶段 3）：可灵 / Runway / fal 上的 I2V 模型，作为"让静态插画动起来"的兜底。

## 5. 依赖清单

已有：`deepagents`、`langchain`、`langgraph`、`langchain-anthropic/openai/openrouter`。需补：

```bash
uv add langchain-mcp-adapters                 # MCP 接入
uv add langgraph-checkpoint-sqlite            # 阶段2 持久化
uv add langchain-chroma langchain-text-splitters  # 阶段2 RAG
uv add fal-client                             # 或 replicate，生图
# 系统依赖：brew install ffmpeg；whisper 可用 mlx-whisper（Mac）或 API
```

环境变量（`.env`）：

```
ANTHROPIC_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ad-video-agent
FAL_KEY=...
```

## 6. 开发路线图

### 阶段 0：地基（≈1 周）
- 跑通 `create_deep_agent` hello world（带一个自定义 @tool）；
- 配好 LangSmith，学会看 trace（这是你调试 agent 的"渲染调试器"）；
- 补课：Python `asyncio` 基础、Pydantic 模型（结构化输出要用）。
- **验收**：CLI 里问 agent 一个问题，它调用你的工具并回答；LangSmith 能看到完整调用链。

### 阶段 1：MVP —— 单代理生成链路（1~2 周）
- 自定义工具 ×2：`parse_script`（LLM 结构化输出 → timeline.json）、`generate_image`（fal/Replicate 封装）；
- `FilesystemBackend` 指向 `projects/`，产物按 4.2 目录规范落盘；
- `skills/` 放第一版 prompt 模板 + 品牌规范；
- `interrupt_on={"generate_image": True}` + `MemorySaver`；
- **验收**：投入一篇文案稿 → 得到 timeline.json + 每镜头 prompt.md + 确认后生成的插画。

### 阶段 2：知识与专业化（2~3 周）
- RAG：Chroma 入库历史项目文档，封装 retriever tool，挂给 `knowledge` 子代理；
- 拆子代理：`prompt-artist`、`illustrator`、`motion-designer`（注意：子代理不继承 skills，需显式传）；
- 动效代码生成（Lottie / Remotion）；ffmpeg + whisper 工具；
- checkpointer 换 SQLite，store 持久化。
- **验收**：问"上次给 X 客户做的风格是什么"能答对；能对插画输出可用的动效代码。

### 阶段 3：交互与集成（2~3 周）
- `langgraph dev` 起本地服务 + 开源 **agent-chat-ui** 做 Web 界面（公司同事可用）；
- 接 MCP（Tavily 搜索、fal MCP）；图生视频工具；视频理解（转写对齐 + 抽帧）。
- **验收**：非技术同事在浏览器里完成一个完整项目流程。

### 阶段 4：部署与评估（持续）
- 部署：LangSmith Deployment（托管，最省事）或自托管 Docker；
- 用 LangSmith datasets 建评估集（好/坏 prompt 样本），回归测试；
- 成本监控、多用户隔离（thread_id / auth）。

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| 生成成本失控 | interrupt_on 人审 + 工具内张数/时长上限 + LangSmith 成本看板 |
| 社区 MCP 质量差 | 核心链路自封装 @tool，MCP 只做外围 |
| 动效自动化预期过高 | 与客户对齐"三级降级"：方案文档保底，代码生成主力，I2V 锦上添花 |
| 需求继续漂移 | Deep Agents 的 skills/tools 都是增量插拔，架构不用推倒 |
| 无技术人员维护 | 知识全部走 Markdown skills；部署用托管；把"加规范"做成写文件而非改代码 |

## 8. 仓库目录结构与语言选择

**语言**：核心逻辑 100% Python。非 Python 部分：skills 是 Markdown（供非技术同事维护）；Web UI 直接 clone 开源 agent-chat-ui（React，只配环境变量不写代码）；可选的 Remotion 动效渲染模板是 Node 脚手架工程（agent 产出 .tsx 放入渲染）；ffmpeg/whisper 为系统工具经 subprocess 调用。

```
Agent-langgraph/
├── pyproject.toml               ★ uv 依赖管理
├── .env / .env.example          ★ API 密钥
├── langgraph.json                 阶段3：langgraph dev/deploy 入口
├── docs/tech-plan-ad-video-agent.md
├── ad_agent/                    ★ Python 包（所有核心代码）
│   ├── __init__.py              ★
│   ├── config.py                ★ .env、模型名、路径、成本上限
│   ├── schemas.py               ★ Pydantic：Shot / Timeline / MotionSpec
│   ├── prompts.py               ★ 主代理与子代理 system prompt
│   ├── agent.py                 ★ create_deep_agent 组装（唯一组装点）
│   ├── cli.py                   ★ CLI 入口（含 interrupt 人审循环）
│   ├── subagents.py               阶段2：子代理定义
│   ├── mcp.py                     阶段3：MultiServerMCPClient 配置
│   ├── tools/                     每个工具一个文件，__init__.py 汇总 ALL_TOOLS
│   │   ├── __init__.py          ★
│   │   ├── script.py            ★ parse_script：文案→Timeline
│   │   ├── image_gen.py         ★ generate_image：fal/Replicate
│   │   ├── retrieval.py           阶段2：RAG 检索
│   │   ├── ffmpeg.py              阶段2：抽帧/裁切/拼接
│   │   ├── transcribe.py          阶段2：whisper 时间戳转写
│   │   └── video_gen.py           阶段3：图生视频
│   ├── rag/
│   │   ├── ingest.py              阶段2：加载→切分→嵌入
│   │   └── store.py               阶段2：Chroma 封装
│   └── graphs/script_pipeline.py  可选：确定性 StateGraph 子图
├── skills/                      ★ 全 Markdown（brand-guidelines / midjourney-prompts / motion-specs / delivery-checklist，每个目录一个 SKILL.md）
├── projects/                    ★ agent 工作区，产物按 4.2 规范落盘
├── knowledge/                     阶段2：RAG 原始语料
├── data/                          阶段2：checkpoints.sqlite、chroma/（gitignore）
├── scripts/                       new_project.py、ingest_knowledge.py
├── tests/                         test_schemas.py、test_tools.py
├── remotion/                      可选：Node 动效渲染模板
└── ui/                            可选：clone 的 agent-chat-ui
```

结构决策：
- 包放仓库根（非 src/ 布局），`uv run python -m ad_agent.cli` 免打包配置直接跑；
- `agent.py` 是唯一组装点，CLI 与 langgraph.json 共用同一 agent 对象；
- `FilesystemBackend(root_dir=".")` 指仓库根，skills/ 与 projects/ 均在 agent 可见范围；
- 代码 / 知识 / 产物 / 运行时数据四区分离，projects/ 与 data/ 进 .gitignore；
- ★ 为阶段 1（MVP）所需的最小集合，其余按阶段增量创建。

## 9. 学习资源（按顺序）

1. **LangChain Academy — Introduction to LangGraph**（免费视频课）：https://academy.langchain.com/courses/intro-to-langgraph
2. **LangChain Academy — Deep Agents 课程**：https://academy.langchain.com （搜 Deep Agents）
3. **Deep Agents 文档**（先读这个树）：https://docs.langchain.com/oss/python/deepagents/overview
   - 工具与 MCP：https://docs.langchain.com/oss/python/deepagents/tools
4. **LangChain 快速上手**（create_agent、@tool、结构化输出）：https://docs.langchain.com/oss/python/langchain/overview
5. **MCP 指南**：https://docs.langchain.com/oss/python/langchain/mcp ；协议本身：https://modelcontextprotocol.io
6. **LangGraph 概览**（需要精细控制流时再深入）：https://docs.langchain.com/oss/python/langgraph/overview
   - Human-in-the-loop（interrupt/resume）、Persistence（checkpointer/store）章节
7. **LangSmith**：https://docs.langchain.com/langsmith/home
8. **RAG 教程**：https://docs.langchain.com/oss/python/langchain/ 下 RAG 章节
9. deepagents 源码与示例：https://github.com/langchain-ai/deepagents
10. Remotion（编程式视频，CG 背景强烈推荐了解）：https://www.remotion.dev/docs
11. Python asyncio 入门：https://realpython.com/async-io-python/

> 学习姿势建议：不要先读完再动手。按路线图边做边查，每做完一步在 LangSmith 里看 trace 理解 agent 到底干了什么——这比读任何文档都快。
