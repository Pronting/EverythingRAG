# T1：RAG 核心框架选型（技术决策）

> 状态：决策草案 v0.1 ｜ 生成日期：2026-08-11 ｜ 承接：`PRD/sections/04-技术架构.md`（§1.2 模块划分、§4 增量同步、§6 搜索演进）
>
> **前置约束（本文件无权推翻，仅细化落地细节与边界）**：Python（FastAPI + asyncio）；Chroma 进程内向量库；bge-m3 本地 ONNX 嵌入；OpenAI 兼容对话模型（默认本地）；方案 B 识图（图片→文字→文本嵌入）；SSE 流式；本地 Web 服务部署；增量同步 = 手动 + 启动检查 + 指纹比对 + 对话级去重 + 语义切块防边界漂移；MVP 只认 MD、纯本地不出机、仅向量检索。
>
> **写作口径**：库的精确版本与具体 API 一律以官方文档为准，本文给的是架构权衡与落地边界，不做版本号断言的承诺。

---

## 0. 决策摘要（TL;DR）

1. **RAG 编排不引框架**：自研轻量管线（约 400–700 行编排胶水），解析用 `markdown-it-py`，嵌入/向量库/模型走自建 Provider 抽象。LangChain/LlamaIndex 作为可选增强备选，不构成硬依赖。
2. **切块 = 解析归库、切块自研**：`markdown-it-py` 解析出 AST，自研「标题锚点分层」切块器（差异化核心，无现成库达标）。
3. **块 ID = `doc_id :: 锚点路径 # 序号`**，`doc_id` 取源路径归一化 hash（内容编辑不改变 doc_id，保证边界稳定）；Chroma 的文档 id 直接用 `chunk_id`，一次 upsert 原子写「向量 + 文本 + 元数据」。
4. **对话处理 = 通用结构特征识别 + 可插拔 FormatAdapter**，平台/时间/标题/角色全进元数据；去重键 = `xxh64(归一化(platform :: conversation_id))`。
5. **增量同步双层指纹**：快路径 stat(mtime,size) 毫秒级全扫；慢路径 xxh64 内容比对；对话级去重键另表维护，文件级指纹与对话级去重键职责分离。
6. **级联删除**：Chroma 按 `where={"doc_id":…}` / `where={"conversation_dedup_key":…}` 批量删，配合 state.db 切块清单保证无孤儿向量。
7. **局部重嵌**：重切后按 `chunk_id + content_hash` 差分，未变块直接跳过嵌入（成本大头在嵌入，跳过即省）。
8. **模块分层**：`api / core / ingestion / models / vectorstore / retrieval / generation / util`，全部外部能力走 `typing.Protocol` 薄接口 + 工厂注册。
9. **关键依赖**（MVP 最小集）：fastapi + sse-starlette + pydantic-settings、markdown-it-py、fastembed(ONNX)、chromadb、openai、xxhash、charset-normalizer、Pillow、aiosqlite；锁版本 + 依赖漏洞扫描。
10. **两点衔接提醒**：① 检索 top-k 的默认值 PRD 4.6（k=5）与技术架构 §6.1（K=20 候选/取 8 入上下文）不一致，需统一（建议 `candidate_k=20` + `context_top_k=5`）；② Chroma 元数据值仅支持 str/int/float/bool，`anchor_path`、`platform=None`、图片路径等须序列化后再写库。

---

## 1. 框架选型：自研轻量管线（不用 LangChain / LlamaIndex 作为硬依赖）

### 1.1 对比

| 维度 | LangChain | LlamaIndex | 自研轻量管线（推荐） |
|---|---|---|---|
| 核心价值定位 | 多 Provider 粘合、Agent 工具生态 | 数据框架：加载器、Node 解析、IngestionPipeline、DocStore | 精确控制入库/增量/去重/块 ID 全流程 |
| 数据接入面 | 大量 loader，但本产品**只认 MD**（既定约束）→ loader 价值≈0 | 加载器丰富，同样与「只认 MD」重叠度低 | 一个 `MDImporter` + 对话 FormatAdapter 就够，自研可控 |
| 切块 | `MarkdownHeaderTextSplitter` 存在但**丢失表格/代码块结构、块 ID 不可自定义** | `MarkdownNodeParser` 不保留「锚点路径+序号」的稳定块 ID | 自研锚点分层切块 = 差异化核心，天然精确 |
| 增量同步/去重 | 无原生概念 | 有 `upsert()/DocStoreStrategy`，但**去重基于文档 hash**，不是「对话内容级」去重键；Node id 由框架分配，无法绑定稳定锚点 | 指纹表/去重键表/切块清单全部自建，正中产品核心诉求 |
| 抽象遮蔽 | 抽象层厚，API 频繁破坏性变更，版本追债重 | 抽象层厚，`llama-index-core` + 一堆可选 extra，依赖树重 | 薄适配层 = 技术架构 §1.2 已定的 Provider 模式，社区可插拔落地最顺 |
| 依赖体积/分发 | 重 | 重（拉入大量 optional 依赖） | 轻：仅解析/嵌入/向量库/HTTP/SQLite |
| 开源后社区贡献适配器 | 贡献者被框架习惯绑定，随框架版本漂移 | 同左 | 一个 `FormatAdapter` 子类 + 注册表即完成，单文件低门槛 |

### 1.2 推荐与理由

**推荐：自研轻量 RAG 管线（自研编排 + 用成熟库做零件），不依赖 LangChain / LlamaIndex 作为硬依赖。**

理由：

1. **差异化核心框架不提供**：增量同步、对话级去重、防边界漂移、稳定块 ID 是本产品区别于通用 RAG 工具的差异化能力，LangChain/LlamaIndex 都没有「对话内容级去重键 + 锚点稳定块 ID」的原生概念。即使采用框架，这些仍需自研，框架只带来额外的抽象税与版本 churn。
2. **数据面被刻意收窄**：「只认 Markdown」是既定决策，直接砍掉了框架最大的价值来源（loader/parser 生态）。三模型均走 OpenAI 兼容单接口（既定决策），Provider 抽象一个 `Protocol` 就能覆盖本地/云端，无需框架的多 Provider 编排。
3. **管线极短**：MVP 入库六步（扫描→解析→切块→嵌入→向量库→状态库）、问答三步（嵌入→检索→生成），不存在需要框架解耦的复杂状态机。400–700 行编排胶水足以表达，且每一行都可测试、可读、可控。
4. **「自研」≠「从零造轮子」**：解析、嵌入推理、向量存储、HTTP、SQLite 全部用战场验证过的库（见 §6），自研的只是「编排 + 差异化逻辑」，完全符合「优先采纳成熟库」的工程原则。
5. **开源可插拔诉求自建方案更顺**：社区贡献者写一个 `FormatAdapter`/`ModelProvider` 子类 + 注册表一行注册即可，不必理解框架内部机制；框架反而增加贡献门槛并随框架版本漂移。

### 1.3 备选与逃逸舱口

- **备选 A（LlamaIndex 作为可选增强，不阻塞 MVP）**：若未来要接大量非 MD 数据源（PDF/网页/Office），可引入 `llama-index-core` 单包，只取它的加载器/文档模型，不进入入库编排主干。前提是自研层的文档模型与切块输出保持「自研优先」，LlamaIndex 仅作为「额外格式导入」的可选转换器。
- **备选 B（LangChain）**：仅在需要 Agent/工具调用链时才考虑；MVP 问答是纯检索生成，无此诉求，**不采用**。
- **逃逸舱口**：自研编排必须把「切块、嵌入、向量库」做成可替换的薄接口（§5），保证若某天框架价值成立，可平移而不重写业务。

### 1.4 边界

- MVP 期间 `pyproject.toml` **不得**声明 `langchain*` / `llama-index*` 依赖（锁死自研路径，避免中途被框架吸引力拖走）。
- 自研编排的每一层必须有单测（chunker 边界用例、去重键归一化用例、指纹快慢路径用例），这是「自研比框架安全」的论证前提。
- 「不用框架」仅限 RAG 编排层；HTTP/校验/解析等基础设施仍必须用成熟库，不得手写。

---

## 2. 切块实现：解析归库、切块自研

### 2.1 Markdown 解析选型

| 方案 | 结论 |
|---|---|
| **markdown-it-py**（推荐） | CommonMark 规范解析器，`renderAsAST()` 直接给结构化节点树，token 含 heading/list/table/fence/code 等类型标注，生态成熟（rich/mdformat 同源），无重型依赖 |
| mistune v3 | CommonMark 规范、更快、有 AST 模式与插件系统；作为备选（若需极致性能或更活跃维护） |
| unstructured | **否决**：为多格式文档设计，MD 只是其一，依赖树重、对「保留精确结构」无优势，与「只认 MD」冲突 |
| 自研正则解析 | **否决**：无法正确识别嵌套代码块、表格、含反引号的 fence、列表缩进等，解析错误会直接污染切块质量与去重正确性 |
| langchain_text_splitters.MarkdownHeaderTextSplitter | **否决**：切块时丢失代码块/表格整体性、块 ID 不可控，只作参考实现 |

**推荐：`markdown-it-py` 解析出 AST → 自研切块器消费 AST。** 解析（通用问题）归库，切块（差异化）自研。

### 2.2 块边界规则（确定性规则，全部有测试用例）

| 元素 | 切块规则 |
|---|---|
| 标题（H1–H6） | **一级边界**：每个标题及其下内容构成一个语义块组；子标题内容归属最近父标题（继承锚点路径） |
| 段落（空行分隔） | 最小切分单元：块内段落 ≥ 1；无标题文档退化为「按段落序切成块」 |
| 代码块（fence/code） | **整体成块，绝不拆散**（满足用户故事「检索到整段代码」） |
| 表格 | **整表成块，绝不拆行**（表格拆行即语义断裂）；超长表格（> 最大长度）整表保留并打 `overlength` 标记，v0.2 再评估 |
| 图片引用 `![](path)` | 文本块内保留引用占位；同时产出独立 `type=image_description` 块（识图开启时，见 §4.3 流程） |
| 列表 | 列表整体入块；块超长时按**顶层列表项**拆分（每项成为子块，继承同一锚点路径） |
| 对话消息 | 每条消息轮次 = 独立块（见 §3） |
| 超长块（> 常量 `MAX_CHUNK_CHARS`，默认 2000） | 块内按「段落 → 句子」逐级拆分，子块继承同一标题路径；代码块/整表例外（不拆） |

**重叠**：块间保留尾部 1–2 句重叠（或拼接标题上下文），重叠块有独立 block_id、互不包含，保证跨块语义不割裂（技术架构 §4.3 已定）。

### 2.3 块 ID 设计（防边界漂移的基石）

```
chunk_id = f"{doc_id}::{anchor_path}#{seq}"
```

- **`doc_id`** = `xxh64(归一化源路径)` —— **必须基于路径而非内容 hash**（细化技术架构 §4.3 示例：`hash(doc)` 会在每次编辑后改变，反而破坏稳定性）。路径移动 → doc_id 变化 → 触发「移除旧路径 + 新增新路径」（对齐 4.5 验收 13 条），行为一致。
- **`anchor_path`** = 标题层级链，如 `/项目笔记/架构决策/技术选型`（`#`/`/` 等保留为链分隔符，允许出现在标题文本中时统一转义）。
- **`seq`** = 锚点组内相对序号（0 起），锚点组内增删段落只改本组 seq，其他锚点组块 ID 不变。
- **`chunk_id` 同时用作 Chroma 文档 id**：一次 `upsert` 原子写「向量 + 文本 + 元数据 + 状态」，更新/删除都按 id 精确命中。

### 2.4 块元数据 schema 草案（写 Chroma 的为标 * 的标量字段）

```jsonc
{
  // —— 通用 ——
  "chunk_id": "{doc_id}::{anchor_path}#{seq}",
  "doc_id": "…",                        // * 路径归一化 xxh64
  "source_file": "绝对路径",              // *（仅存元数据，不拷贝原文）
  "doc_type": "document" | "conversation",// *
  "block_type": "heading|paragraph|code|table|list|image_description", // *
  "anchor_path": "/项目笔记/架构决策/技术选型",  // * 序列化为字符串（Chroma 元数据不支持数组）
  "seq": 3,                              // *
  "content_hash": "xxh64(归一化块文本)",    // *（差分重嵌依据）
  "char_count": 1234,                    // *
  "updated_at": "ISO8601",               // *
  // —— 对话专属（doc_type=conversation 时）——
  "platform": "chatgpt|kimi|deepseek|unknown", // *（未知平台用空串，不用 None）
  "conversation_title": "…",              // *
  "conversation_time": "ISO8601",        // *
  "message_role": "user|assistant|system|tool", // *
  "message_seq": 7,                      // *
  "conversation_dedup_key": "…",         // *（供按对话级联删除/过滤）
  // —— 图片专属（block_type=image_description）——
  "image_path": "原图绝对路径",            // *
  "image_description": "VLM 描述文本"      // *（仅库内使用）
  // —— 链接（只记录不抓取）——
  // "outgoing_links" 等非标量字段不入 Chroma，存 state.db chunks 表按 chunk_id 关联
}
```

> **Chroma 元数据约束**：值仅支持 `str / int / float / bool`，`None` 与数组不受支持 → `anchor_path`、平台未知、图片路径清单、外链清单都须序列化（字符串/拼接）后再写库；「未启用识图」的图片引用不进向量库，只留元数据（对齐 4.3 验收 2）。

### 2.5 边界 / 依赖

- 最大块长度、重叠长度、候选 K、置信阈值全部收敛为配置常量（`config/chunking.py`），禁止魔法数。
- 超长表格、无标题纯笔记的「允许小概率局部漂移」是 MVP 接受的折衷，写入切块器 docstring 与测试预期（技术架构 §4.3 已声明）。
- 依赖：`markdown-it-py`（解析）、（可选）`mdit-py-plugins` 的 front-matter 插件读取 YAML 元数据头。

---

## 3. 对话文档处理：通用结构识别 + 可插拔 FormatAdapter

### 3.1 识别策略（不逐平台写死，对齐 4.1「通用对话结构特征识别」）

**判定为对话文档**（满足其一）：
- 出现 ≥2 组**重复的角色标记轮次**（如 `**user:**` / `**assistant:**` 加粗前缀行、列表项 `- user: …`、引用块 `> user:`、JSONL/JSON 结构 `{"role":"user","content":…}`）。
- 同时出现 ≥1 个 user 轮 + ≥1 个 assistant 轮。

**平台推断优先级**：文件名/目录名特征（chatgpt/kimi/deepseek/claude…）→ 格式指纹（各平台导出的标记风格特征）→ 兜底 `unknown`（导入后允许用户手动补充，对齐 4.1 验收 6）。

**FormatAdapter（可插拔）**：

```python
class FormatAdapter(Protocol):
    name: str                      # 平台名，如 "chatgpt"
    def can_handle(self, filename: str, text_head: str) -> float: ...   # 置信度 0~1
    def parse(self, text: str) -> Conversation: ...                      # 归一化→轮次→元数据
```

- 默认内置 1 个「通用适配器」（识别通用角色标记轮次）+ 可注册的按平台精化适配器；注册表按置信度取最高者，无注册表负担。
- 新增平台 = 新写一个适配器类 + `register_platform_adapter(...)` 一行注册。

### 3.2 解析策略

1. **归一化**（Importer 层）：把各平台导出文本归一为标准 MD 纯文本（统一换行、剥离导出噪音/时间戳前缀），产出标准化文本 + 候选平台。
2. **轮次切分**：扫描角色标记，每条消息正文延伸到下一个角色标记（或空行分隔歧义处用「最长内容归属」规则）。支持 system/tool 角色（保留但可配置是否入库）。
3. **元数据提取**：导出时间或对话开始时间（取可用者）、对话标题（导出内嵌标题 → 文件名茎 → 首条消息摘要 三级退化）、角色序列。
4. **切块**：每条消息 = 独立块；单条消息超长按段落/句子再拆，子块继承 `message_seq` 与角色。

### 3.3 元数据 schema

见 §2.4 的「对话专属」字段（`platform / conversation_title / conversation_time / message_role / message_seq / conversation_dedup_key`）。`Conversation` 领域对象：

```python
@dataclass
class Conversation:
    platform: str
    title: str
    time: datetime | None
    messages: list[ConversationMessage]   # (role, seq, text)
    identity: str                          # 用于去重键的会话身份，见 §4.2
    source_file: str
```

### 3.4 边界 / 依赖

- 不做 token / API / 浏览器抓取；外链只进元数据，不发任何网络请求（既定约束）。
- 识别为「未知平台」时必须能进库、可手动补平台，不能因平台未知阻塞导入。
- 角色序列在 `state.db` 持久化，用于按对话展示「角色时间线」与来源标注。
- 依赖：`markdown-it-py`（复用解析）、`charset-normalizer`（编码检测，对齐 4.1 验收 12）、`xxhash`（去重键）。

---

## 4. 增量同步实现（指纹双层 + 对话去重键 + 级联删除 + 局部重嵌）

### 4.1 指纹比对（快慢路径）

`state.db` 表：`files(path PK, mtime_ns, size, content_hash, last_ingested_at, status)`

- **快路径**（启动检查 + 手动同步的全库扫描，只 stat 不读文件）：`(mtime_ns, size)` 均未变 → 跳过。目标 10 万文件 < 3s（技术架构 §7）。
- **慢路径**：快路径命中「有变化」→ 读文件算 `content_hash`（`xxhash.xxh64`，冲突概率对本量级可忽略；备选 stdlib `hashlib.blake2b`）：
  - 未变（仅 touch）→ 只更新 `mtime_ns`，不重处理；
  - 变了 → 标记重入库；
  - 库中存在但磁盘缺失 → 标记删除。
- **失败兜底**：读失败/编码失败 → 记失败清单，UI 可单独重试，不中断整体同步（对齐 4.5 验收 11）。

### 4.2 对话级去重键（与文件指纹职责分离）

- **文件级指纹**判断「文件变没变」；**对话级去重键**判断「是不是同一段对话」——两者独立、必须同时维护（技术架构 §4.1/§4.2 已定）。

```
dedup_key = xxh64(normalize(platform + "::" + conversation_identity))
```

- `conversation_identity` 解析优先级：导出内嵌会话 ID/标题 → 文件名茎（清洗）→ 退化「首条消息归一化内容 + 消息条数」组合。
- `normalize`：统一换行（CRLF→LF）、折叠多余空白/空行、剥离线级时间戳噪音、可选降级大小写——抵抗「仅格式变化」误判。
- `state.db` 表：`conversations(dedup_key PK, platform, title, file_path, last_seen, status, first_seen_at, content_hash)`。
- **过期删除**：每轮同步把命中的 `dedup_key` 打 `last_seen`；同步末清理本轮未被引用的键 → 判定对话被删 → 级联删其块。同一文件内多次出现同一对话：以第一次为准，后续刷新 `last_seen`（技术架构 §4.2）。
- **验收对齐**：平台维度保留（两个平台相似对话不合并，4.5 验收 10）；换文件名/换导出格式去重键不变（验收 7/8）；手动编辑内容仍识别同一对话并按增量更新处理（验收 9）。

### 4.3 文件→块→向量级联删除

- `state.db` 切块清单：`chunks(chunk_id PK, doc_id, conversation_dedup_key, content_hash, anchor_path, seq, status)` 维护「文件/对话 → 块 ID」映射（技术架构 §4.3 要求的锚点→块ID→内容hash→向量ID 映射即此表）。
- **删除文件**：`chunks` 查该 `doc_id` 全部 `chunk_id` → `chroma.collection.delete(ids=[...])`（或按 `where={"doc_id": …}` 批量删）→ 删 `chunks` 行 → 删 `files` 行。
- **删除对话**：`chunks` 查该 `conversation_dedup_key` 全部块 → `collection.delete(where={"conversation_dedup_key": …})` → 删行 → 删 `conversations` 行。
- **幂等**：事务边界在「文件级」，先处理后提交状态；单文件失败不污染其他文件（技术架构 §7 可靠性）。

### 4.4 局部修改只重嵌受影响块

重入库一个「已变化」文件时：
1. 重新解析 + 切块，得到新块清单；
2. 与旧 `chunks` 清单按 `chunk_id` 差分：
   - `chunk_id` 存在且 `content_hash` 未变 → **跳过嵌入**（复用旧向量与元数据）；
   - `chunk_id` 存在但 `content_hash` 变 → 重新嵌入 + `upsert(ids=[chunk_id], …)`（原子覆盖）；
   - 新 `chunk_id` → 嵌入 + upsert；
   - 旧清单有、新清单无 → 删除（锚点被删，只删该锚点下块）。
3. 嵌入是入库成本大头，差分命中即省（技术架构 §4.3）。

### 4.5 边界 / 依赖

- 不做实时监听（MVP，既定约束）；同步只由「手动按钮 + 启动检查」触发。
- `content_hash` 用 xxh64，本量级足够；若未来量级暴涨换 blake2b 只需改一个函数。
- 路径移动按「移除旧 + 新增新」处理并提示（对齐 4.5 验收 13），避免路径漂移静默重复。
- 依赖：`xxhash`、`sqlite3`（stdlib，WAL 模式；或 `aiosqlite` 异步封装，见 §6）。

---

## 5. 模块分层与接口边界（保持可插拔）

### 5.1 目录结构（对齐技术架构 §1.2 模块划分）

```
everything_rag/
├── api/                        # FastAPI 层：路由 + SSE + 静态托管
│   ├── app.py                  # create_app()：挂路由、加载配置、启动检查
│   ├── routers/                # ingest / sync / chat / models_config / kb_admin
│   └── schemas.py              # pydantic 请求/响应 DTO
├── core/                       # 领域核心
│   ├── config.py               # AppConfig（pydantic-settings）
│   ├── state.py                # state.db（SQLite，WAL）：files/conversations/chunks 仓储
│   ├── sync.py                 # SyncOrchestrator：扫描→差分→删除→状态提交
│   └── errors.py               # 领域异常
├── ingestion/
│   ├── scanner.py              # 目录扫描 + 指纹快慢路径
│   ├── importer.py             # MD 读取/编码检测/归一化/失败分类
│   ├── dialogue/               # detector / segmenter / adapters（FormatAdapter 注册表）
│   └── chunker/                # md_ast.py（markdown-it-py→节点树） / chunker.py / block_ids.py / metadata.py
├── models/
│   ├── embedding/              # base(Protocol) / fastembed_engine / factory
│   ├── chat/                   # base(Protocol) / openai_compat / factory
│   └── vision/                 # base(Protocol) / openai_compat / factory
├── vectorstore/
│   ├── base.py                 # VectorStore(Protocol)
│   ├── chroma_store.py
│   └── factory.py
├── retrieval/
│   ├── retriever.py            # 问题嵌入 → 余弦 top-K → 元数据过滤 → 置信阈值
│   └── scoring.py
├── generation/
│   ├── prompt.py               # 上下文组装 + 强制来源标注指令
│   ├── qa.py                   # 检索→组装→生成 编排（含无结果/低置信分支）
│   └── streaming.py            # SSE 流式适配（流出口抽象，预留 WebSocket）
└── util/
    ├── hashing.py              # xxh64
    ├── normalize.py            # 文本/去重键归一化
    └── logging.py              # 脱敏日志过滤器
```

### 5.2 关键接口签名草案（`typing.Protocol`，全部可替换、可 mock）

```python
# vectorstore/base.py
class VectorStore(Protocol):
    def upsert(self, ids: list[str], embeddings: list[list[float]],
               documents: list[str], metadatas: list[dict]) -> None: ...
    def delete(self, ids: list[str] | None = None, *, where: dict | None = None) -> None: ...
    def query(self, embedding: list[float], top_k: int, *,
              where: dict | None = None) -> QueryResult: ...   # 返回 (ids, distances, metadatas, documents)
    def get_by_ids(self, ids: list[str]) -> list[ChunkRecord]: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...

# models/embedding/base.py
class EmbeddingModel(Protocol):
    name: str
    dim: int
    def embed_texts(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

# models/chat/base.py
class ChatModel(Protocol):
    async def stream(self, messages: list[Message], *, params: GenParams) -> AsyncIterator[str]: ...

# models/vision/base.py
class VisionModel(Protocol):
    async def describe(self, image_path: str, prompt_template: str) -> str: ...

# ingestion/dialogue/adapters/base.py
class FormatAdapter(Protocol):
    name: str
    def can_handle(self, filename: str, text_head: str) -> float: ...
    def parse(self, text: str) -> Conversation: ...

# ingestion/chunker/chunker.py
def chunk_document(ast: MDNode, *, doc_id: str, source_file: str) -> list[Chunk]: ...
def chunk_conversation(conv: Conversation, *, doc_id: str) -> list[Chunk]: ...

# core/sync.py
class SyncOrchestrator:
    def scan(self, root: Path) -> SyncPlan: ...          # 快路径，只 stat
    def ingest(self, plan: SyncPlan) -> SyncReport: ...   # 慢路径 + 差分重嵌 + 级联删除
    def reconcile_deletions(self, plan: SyncPlan) -> None: ...
```

- **并发模型**：入库管线（解析/嵌入/向量写）为 CPU/IO 密集 → `asyncio + run_in_executor(threadpool)`；问答流式走 `async` 直连（MVP 单用户，无多进程诉求，对齐技术架构 §3.1）。
- **惰性加载**：嵌入模型首次检索/入库才加载（对齐技术架构 §7 冷启动 < 5s）。
- **出网收敛**：唯一允许外呼的两个出口 = `ChatModel` / `VisionModel` 的 Provider 实现（技术架构 §5.2），代码审查锚点。

### 5.3 边界

- 业务层禁止直接拼 Chroma SQL/where 或 sqlite SQL——一律经 `VectorStore` / `StateRepo` 接口（技术架构 §3.2 已强制）。
- 所有接口先有 mock/fake 单测（`FakeVectorStore`、`FakeEmbeddingModel`），保证可插拔性被测试锁住。

---

## 6. 关键依赖清单（MVP 最小集）

> 版本策略：`pyproject.toml` 用 `~=` 下限约束 + lockfile 精确保留（uv 或 pip-tools 生成），开源分发前跑依赖漏洞扫描（技术架构 §5.4 已定）。下表具体版本号**以官方文档为准**。

| 包 | 用途 | 版本策略 |
|---|---|---|
| `fastapi` | API + `StreamingResponse` | 当前 stable 大版本 |
| `uvicorn[standard]` | ASGI 服务器，`127.0.0.1:随机端口` | 同左 |
| `sse-starlette` | SSE 流式封装（或直接用 `StreamingResponse` + `text/event-stream`） | 轻依赖，二选一 |
| `pydantic` + `pydantic-settings` | 配置校验、请求/响应 DTO、`.env` 注入 | 2.x |
| `markdown-it-py` | CommonMark 解析 → AST（切块输入） | 3.x |
| `mdit-py-plugins` | front-matter（YAML 元数据头）插件 | 随 markdown-it-py |
| `fastembed` | bge-m3 本地 ONNX 嵌入（免 torch，依赖 `onnxruntime`） | 以官方支持模型列表为准；备选 `sentence-transformers`+`optimum`（更重，需 torch） |
| `chromadb` | 进程内持久化向量库（自带嵌入默认关闭，用我们传入的向量） | 以官方当前 API 为准（`PersistentClient`）；启动时设 `ANONYMIZED_TELEMETRY=False` 关遥测保隐私 |
| `openai` | OpenAI 兼容 `AsyncOpenAI(base_url=…)`，同时服务本地 Ollama 与云端 | 1.x |
| `xxhash` | xxh64 内容指纹 / 去重键 | 3.x；备选 stdlib `hashlib.blake2b` |
| `charset-normalizer` | 非 UTF-8 编码检测（对齐 4.1 验收 12） | 当前 stable |
| `Pillow` | 图片解码 / EXIF 方向修正 / 降采样（方案 B 识图前置） | 11.x |
| `aiosqlite` | state.db 异步访问（或 stdlib `sqlite3` 走线程池，二选一） | 当前 stable |
| `pytest` + `pytest-asyncio` | 测试（开发期，非运行期依赖） | 当前 stable |

**明确不引入**：`langchain*`、`llama-index*`（§1）、`unstructured`（§2.1）、`transformers`/`torch`（MVP 嵌入用 fastembed/ONNX 免重型栈；v0.2 如需 bge-m3 稀疏向量分支再评估 `FlagEmbedding` 或 ONNX 稀疏路径）、`rank_bm25`（v0.2 混合检索再引入）、`bge-reranker`（v1.0 再引入，走 `sentence-transformers`）。

---

## 7. 风险与待对齐项

| # | 项 | 说明与建议 |
|---|---|---|
| 1 | 检索 top-k 默认值冲突 | PRD 4.6 验收 1「top-k 默认 5」vs 技术架构 §6.1「K=20 候选/取 8 入上下文」。建议统一为 `retrieval.candidate_k=20`、`retrieval.context_top_k=5`（对齐 PRD），在 PRD 修订时标注 |
| 2 | 置信阈值 | `min_score`（余弦）MVP 默认 0.30–0.40，须用作者真实语料在 v0.1 校准；无结果/低置信分支须显式提示而非硬凑 |
| 3 | Chroma 元数据值类型限制 | `anchor_path`、`platform=None`、图片/外链清单需序列化；用「只把过滤需要字段写 Chroma、其余写 state.db 按 chunk_id 关联」的双写策略 |
| 4 | Chroma 版本 API 迁移 | 0.5.x→1.x Python 客户端接口有调整，切块/去重逻辑与 `VectorStore` 接口解耦已覆盖此风险，升级只动 `chroma_store.py` |
| 5 | 超长表格 | MVP 整表保留 + `overlength` 标记，v0.2 评估行级拆分或表格摘要 |
| 6 | 无标题纯笔记边界漂移 | MVP 接受该文档内小概率局部漂移（技术架构 §4.3 已声明），v0.2 评估滑动窗口重叠补偿 |
| 7 | 依赖供应链 | lockfile + 分发前漏洞扫描；模型权重许可逐项审计（技术架构 §3.5 已定，默认捆绑模型必须宽松许可） |
| 8 | 重建索引并发 | 重建期间旧索引保持可读或显式不可用提示，二选一须在实现期定并落验收（4.4 验收 5） |
