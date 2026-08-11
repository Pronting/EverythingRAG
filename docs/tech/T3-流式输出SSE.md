# T3 流式输出（SSE）技术选型细节与边界

> 承接：PRD `04-技术架构.md` §3.4「对话模型与流式输出（SSE 推荐）」、§1.3 问答数据流；`03-功能需求.md` §4.6 问答 Dashboard；`05-范围与路线图.md` §8.1 v0.1 验收/性能指标。
> 状态：待并入技术选型决策文档 ｜ 生成日期：2026-08-11
> 前置约束（本子代理无权推翻）：SSE（FastAPI `StreamingResponse` + 前端 `EventSource`），不用 WebSocket（MVP）；对话模型 OpenAI 兼容可插拔（默认本地 Ollama/llama.cpp）；纯本地、数据不出机；多轮对话；答案强制来源标注；中文为主。本文件只细化落地细节与边界，不推翻既定方向。

---

## 0. 决策摘要

| # | 决策点 | 推荐 | 备选 | 边界（MVP 不做） |
|---|---|---|---|---|
| 1 | SSE 实现 | `StreamingResponse(media_type="text/event-stream")` + 自定义命名事件帧（`meta`/`token`/`done`/`error`），不用 sse-starlette | sse-starlette `EventSourceResponse` | 事件流内不做鉴权/二进制帧；不做流内插帧 |
| 2 | 前端消费 | 原生 `EventSource`（GET 端点 + query 参数），逐 token `append` 到 `textContent` | fetch + ReadableStream + AbortController 手写 SSE 解析（升级路径） | 不做流内 Markdown 实时重渲染 |
| 3 | 取消/中断 | 「停止」= `EventSource.close()` 断开连接 → FastAPI 取消生成器 → 关闭上游模型流；`request.is_disconnected()` 早停 | DELETE /api/chat/stream/{id} 取消注册中的 in-flight 任务 | 中断后不续跑；不保留可重启的上游状态 |
| 4 | 断线/重连 | `retry: 3000` 初始行 + `: hb` 心跳注释帧；**不做服务端续传**，重连=标记中断+保留已流出内容+用户重试 | 服务端用 `id:`+`Last-Event-ID` 实现续传（v1.0 评估） | 不实现断点续传/去重重放 |
| 5 | 多轮上下文 | 会话内存态消息数组，最近 N 轮（默认 10，可配）+ 字符预算截断；检索用当前问题（追问退化为「上一次答案」作查询） | 会话持久化到 SQLite / 摘要压缩历史（v0.2） | 历史不做摘要压缩；不做跨会话记忆画像 |
| 6 | 错误与指标 | 事件类型区分数据/错误（`error` 事件 + 错误码）；`done` 事件携带 `ttft_ms`/`tok_per_s`/字数；进程内指标环缓冲 + `/api/metrics` | 结构化日志（request_id） | 不接外部可观测系统；日志不含正文/密钥 |
| 7 | 总体边界 | 客户端中途改请求、服务端主动推送、WebSocket 叠加、流内来源穿插均不做 | 升级信号见 §7.4 | — |

---

## 1. SSE 实现方案

### 1.1 推荐：FastAPI `StreamingResponse` + 自定义命名事件帧

- **理由**：SSE 是「服务器向单个 HTTP 连接单向写文本」的纯文本协议，`StreamingResponse` 原生支持任意 `async generator` 作为 body，事件帧格式用字符串拼装即可，零额外依赖、对编码/心跳/事件命名完全可控。sse-starlette 提供的 `EventSourceResponse` 主要价值是 `ping` 心跳与 `Last-Event-ID` 自动回显——本项目 MVP 明确不做续传（见 §4），且本地回环场景心跳需求弱，故不引入该依赖。
- **接线方式**：流式出口封装为一层 `StreamOutput` 抽象（对齐 `04-技术架构.md` §3.4「流式出口做一层 StreamOutput 抽象」）。上游对话模型统一产出**字典帧**（`{"type": ...}`），SSE 是当前唯一的帧序列化器；未来若要叠加 WebSocket，同一帧流可直接换一个序列化器，不推翻生成侧代码。

### 1.2 事件流格式设计（协议规范，前后端契约）

每条帧以空行结尾；`data` 一律为 **JSON 字符串**（UTF-8）。事件命名：

| 事件 | 方向 | data 负载（JSON） | 语义 |
|---|---|---|---|
| `meta` | 首个事件，开始生成前 | `{"type":"meta","request_id","sources":[{id,file,path,platform,title,relevance}],"no_hit":false}` | 一次性下发检索来源卡片 + 低置信标记，避免流中穿插来源 |
| `token` | 每 chunk | `{"type":"token","text":"<完整字符串>","seq":123}` | 增量文本，**必须为完整的 Unicode 字符串**（见 §1.4） |
| `done` | 结束（正常/无命中） | `{"type":"done","request_id","outcome":"ok|no_hit","message?","ttft_ms","tok_per_s","total_chars","total_tokens"}` | 正常收尾；`no_hit` 时无 token、附 `message`（如「未检索到足够相关内容」） |
| `error` | 中止 | `{"type":"error","code":"model_not_configured|upstream_error|upstream_timeout|no_hit|internal","message","partial":true}` | 生成失败/配置缺失；`partial=true` 表示已流出部分内容 |
| （默认 `message`） | — | 不使用 | 全部走命名事件，避免默认事件与业务数据混淆 |
| 心跳注释帧 | — | `: hb`（注释行） | 仅保活，不触发任何前端事件处理器 |

**结束标记**：不使用 OpenAI 风格的 `data: [DONE]`。上游 `[DONE]` 是 OpenAI 兼容服务内部约定，我们在透传层消费掉、翻译成自己的 `done` 事件。这样前端只需要监听 `done`/`error` 两种终态，语义清晰。

帧构建器示意（核心，帧格式遵循 SSE 规范：`event:`/`data:`/空行，多行 `data:` 用 `\n` 连接）：

```python
# sse.py —— StreamOutput 抽象 + 帧序列化
from fastapi.responses import StreamingResponse

def sse_frame(event: str, payload: dict) -> str:
    import json
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

class StreamOutput:
    """把『生成器产出的字典帧流』序列化为 SSE HTTP 响应。未来换 WebSocket 只换这里。"""
    def __call__(self, frames) -> StreamingResponse:
        return StreamingResponse(
            self._wire(frames),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # 防未来反代缓冲破坏流式；本地直连时无害
                "Connection": "keep-alive",
            },
        )

    @staticmethod
    async def _wire(frames):
        # 首行：重连间隔；随帧带心跳注释，保证空闲连接不被中间层掐断
        yield "retry: 3000\n\n"
        while True:
            try:
                frame = await frames.__anext__()
            except StopAsyncIteration:
                return
            if frame is None:            # None = 心跳占位
                yield ": hb\n\n"
                continue
            yield sse_frame(frame["type"], frame)
```

### 1.3 上游 OpenAI 兼容 `chat/completions` stream 的逐 token 透传/转换

推荐用官方 `openai` SDK 的 `AsyncOpenAI`（本地 Ollama/llama.cpp 都暴露 OpenAI 兼容 `/v1/chat/completions` 端点，`base_url` 指向 `http://127.0.0.1:<port>/v1` 即可）。透传层只做一件事：**把上游 chunk 的 `delta.content` 解码成完整 Python `str` 后，包装成 `token` 帧**。`[DONE]`、`usage` 等由 SDK/逐行解析消费，不落到下游。

```python
# providers/chat/openai_compat.py（示意；具体以 openai SDK 当前文档为准）
from openai import AsyncOpenAI

async def stream_chat(messages, *, base_url, api_key, model,
                      temperature=0.7, max_tokens=None, timeout=60.0):
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, stream=True,
            temperature=temperature, max_tokens=max_tokens,
            # 部分兼容端点支持 usage；不支持时忽略即可（见 §6）
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if not chunk.choices:
                continue                       # usage / finish 等非增量 chunk
            delta = chunk.choices[0].delta.content
            if delta:                          # 仅转发有内容的增量
                yield {"type": "token", "text": delta}
    finally:
        await client.close()                   # 取消/异常时也必须关掉上游连接
```

**备选**（若不想依赖 SDK，或需要对原生字节做精细控制）：用 `httpx.AsyncClient.stream("POST", url, json=...)` 逐行读 `data:` 行自行解析 SSE，遇到 `data: [DONE]` 停。SDK 更省事、社区维护好，作为推荐；裸 httpx 更透明、依赖更少，作为备选。两者产出的都是同一个 `{"type":"token"}` 帧流，路由侧无感知。

### 1.4 UTF-8 中文流式编码注意事项（防乱码/半字）

- **服务器侧铁律：`data` 永远是完整的 Unicode 字符串，绝不分帧发送多字节字符的半截字节。** SSE 帧是 UTF-8 文本，浏览器对每一帧独立做 UTF-8 解码；若某帧末尾是被截断的多字节序列（例如某个中文字符的第 1 个字节在第 n 帧、后 2 字节在第 n+1 帧），浏览器会解出 `U+FFFD` 替换符（乱码/「半字」）。
- **为什么本方案天然规避**：LLM 的 `delta.content`（Qwen/Ollama 等）解码后就是完整 `str`；`json.dumps(..., ensure_ascii=False)` 输出的是完整码点串；`StreamingResponse` 以 UTF-8 编码整帧写出。整条链路没有「按字节切」的环节，因此**不会出现半字**。
- **保留兜底策略**：若将来某个 Provider 只能给原生字节流（byte-level tokenizer），前端需用 `TextDecoder("utf-8", { stream: true })` 做增量解码（它会缓存半截序列、等后续字节到齐再解出完整字符）；本方案用 JSON 字符串传输时该兜底不需要。
- **中文 token 粒度**：中文常用字在 Qwen 系 tokenizer 中多为 1–2 token，单 token 增量可能只有 1 个汉字甚至半个词；前端按「字符」追加渲染即可，无需按 token 单位处理。
- **前端渲染层面**（见 §2.2）：逐 token 期间**只 append 纯文本**（`textContent`），不做整段 Markdown 重渲染，避免「代码围栏/行内代码/链接尚不闭合时反复闪烁」；Markdown 化仅在 `done` 后（或 debounce 后）执行一次。

---

## 2. 前端消费

### 2.1 推荐：原生 `EventSource` + GET 端点

- **用法定式**（原生浏览器 API，无 polyfill）：

```javascript
// chat_stream.js（示意）
const es = new EventSource(`/api/chat/stream?q=${encodeURIComponent(question)}&conv=${encodeURIComponent(convId)}`);

es.addEventListener('meta', (e) => {
  const m = JSON.parse(e.data);
  if (m.no_hit) { renderNoHit(m.message); es.close(); return; }
  renderSources(m.sources);              // 来源卡片（先渲染，答案再流式进来）
  resetAnswerBlock();
});

es.addEventListener('token', (e) => {
  const t = JSON.parse(e.data);
  appendToken(t.text);                   // 见 §2.2 增量渲染
});

es.addEventListener('done', (e) => {
  const d = JSON.parse(e.data);
  if (d.outcome === 'no_hit') renderNoHit(d.message);
  finalizeAnswer();                      // Markdown 渲染 + 指标展示
  es.close();                            // 主动关闭，阻止自动重连
});

es.addEventListener('error', (e) => {
  // 注意：error 事件在『连接失败』与『服务端主动关闭』时都会触发。
  // 正常流程我们已经在上面的 done 里 es.close()，不会走到这里。
  if (es.readyState === EventSource.CLOSED) return;  // 主动关闭
  // 走到这里 = 传输层异常或服务端在未发 done 时断开 → 视为中断，见 §4
  onInterrupted();
});
```

- **关键约束与解法**：`EventSource` **只支持 GET**，无法携带 JSON body。因此流式端点定义为 `GET /api/chat/stream?q=...&conv=...`（本地回环、问题通常 <1KB，query 长度安全；隐私上 query 只落本机访问日志，见 `04-技术架构.md` §5.3 脱敏要求）。这符合既定方向（前端 `EventSource`）且零 polyfill。
- **无需 polyfill**：现代浏览器（Chrome/Edge/Firefox/Safari）全部原生支持 `EventSource`、`ReadableStream`、`AbortController`。MVP 不追求 IE/旧内核。
- **升级路径（不做，但留好接口）**：当需要 POST body / 读 HTTP 状态码 / 更干脆的取消时，可用 `fetch` + `response.body.getReader()` + `TextDecoder` 手写一个约 40 行的 SSE 行解析器，配 `AbortController`。因为服务端仍是同一份 SSE 字节流（`text/event-stream`），前端换实现**不影响后端任何代码**。

### 2.2 增量渲染方案

- **流式期间**：维护一个「当前答案块」的 DOM 文本节点，`appendToken` 用 `node.textContent += text` 或 `document.createTextNode` 追加。**绝不使用 `innerHTML` 直插**——模型输出与检索上下文均为不可信文本，违反会直接撞上 `04-技术架构.md` §5.4 的 XSS 红线。由于只追加文本节点，浏览器无需重排整个消息块，滚动位置不跳动。
- **完成后**：用「白名单 Markdown 渲染器 + HTML 消毒」把整段答案转成富文本（如 `marked` + `DOMPurify`，允许的标签/属性白名单化），一次性渲染替换纯文本块。避免每 token 重渲染导致的闪烁与光标跳动。
- **中文增量**：逐 token 追加字符，天然无半字（服务端保证整码点帧，见 §1.4）；若期间检测到输入法组合（`compositionstart`）等情况，只影响用户输入框，与答案流无关。
- **来源卡片**：`meta` 事件一次性拿到 `sources`（文件/对话/平台/图片缩略图路径），在答案流开始前就渲染卡片、支持点击跳转原文——对应验收「来源标注可点击核验」。

---

## 3. 取消与中断

### 3.1 推荐：断开连接即取消 + 连接断开检测

链路：用户点「停止」→ `es.close()` → 浏览器断开 TCP → FastAPI 侧检测到客户端断开 → `StreamingResponse` 关闭 async generator → 在 `finally` 里关闭上游模型连接。全程零额外状态、零轮询，符合 `04-技术架构.md` §3.4「SSE 连接断开即停止生成」。

路由与生成器的取消骨架：

```python
# api/chat.py（示意）
from fastapi import Request
from fastapi.responses import StreamingResponse

@app.get("/api/chat/stream")
async def chat_stream(request: Request, q: str, conv: str = ""):
    # ...前置：校验配置（缺对话模型 → 直接产 error 帧）、检索（产 meta 帧）...

    async def frames():
        try:
            async for tok in provider.stream_chat(messages, **cfg):
                if await request.is_disconnected():   # 早停：客户端已走，不再等上游
                    return
                yield {"type": "token", "text": tok}
            yield {"type": "done", "request_id": rid, **metrics()}
        except asyncio.CancelledError:
            # 客户端断开/服务端关闭 → anyio 取消注入到生成器
            raise
        finally:
            await upstream_cleanup()   # 关闭 httpx/AsyncOpenAI 连接，见 §1.3 finally

    return StreamOutput()(frames())
```

要点：

- **`request.is_disconnected()`** 在每帧之间轻量探测，客户端已断开时提前 `return`，不必等上游模型吐完或等下一次 yield 才被取消。
- **`finally: await upstream_cleanup()`** 保证无论正常结束、异常、还是被取消，上游 HTTP/模型连接都被关闭，不泄漏本地 Ollama 的 socket/连接。
- **取消传播**：本地模型（Ollama）收到连接关闭即停止推理，占用的显存/内存立即释放；这是「用户点击停止 = 省算力」的即时性来源。

### 3.2 备选：服务端 in-flight 任务注册表 + 取消端点

`DELETE /api/chat/stream/{request_id}` → 后端持有一个 `{request_id: asyncio.Task}` 注册表，`task.cancel()`。仅当「断开检测不可靠」或「同一会话多端并发」时才有价值；MVP 本地回环单用户场景不需要，列为备选。

### 3.3 边界

- **MVP 不做「中断后续跑」**：中断 = 生成任务已不可恢复，客户端保留已流出内容 + 提供「重试」按钮（重试 = 重新发起一次完整请求），对应验收 9「保留已流出的部分，提示错误并允许重试，不静默截断」。
- **不做流内用户指令**：用户不能在答案生成中途插入/修改请求（如「停！换个角度」），那是 WebSocket 双向通道的诉求，见 §7。

---

## 4. 断线与重连

### 4.1 推荐：`retry` 初始行 + 心跳注释帧；重连 ≠ 续传

- **重连机制**：`EventSource` 连接失败会自动重连（浏览器内置），重连间隔由服务端 `retry: 3000`（毫秒）控制（见 §1.2 帧构建器首行）。服务端空闲时定期吐 `: hb` 注释帧（可配 15–30s），防止本地反向代理/系统网络栈在「首 token 前的思考期」把连接当死连接掐掉。注释帧不触发任何前端事件处理器（`onmessage`/自定义事件都不会收到），是 SSE 官方约定俗成的保活写法。
- **为什么 MVP 不做服务端续传**：LLM 生成是「有状态的有序流」，断点续传要么服务端缓存已生成 token（内存+复杂度），要么重新生成（浪费）。本地个人场景断线概率极低，为它引入 `id:`/`Last-Event-ID` 续传协议与去重重放逻辑不划算。**决策：断线即中断**。

### 4.2 重连后的行为（防「重复插入半截内容」）

规则：客户端维护 `receivedDone` 标志。

1. `done` 事件到达 → `es.close()` → 不会再重连 → 无重复。
2. 传输层断开且未收到 `done`（触发 `error` 事件或 `es.readyState === CONNECTING` 表示重连中）：
   - **立即把当前答案块标记为「已中断」**，保留已流出的部分文本（不删除、不截断）。
   - 展示错误条「生成中断，可点击重试」+「重试」按钮。
   - 用户在 `open` 事件（重连成功触发）时若 `!receivedDone`，也按「中断」处理，**不自动重新生成**。
3. 「重试」= 用户显式触发：**新建一个答案块替换中断块**（清空/标记旧块），再发起一次新的流请求。因为旧流不会续传，新流从零开始，**不会出现两段内容拼接重复**。

> 关键点：不自动重连续跑，也就天然避免了「断线重连后把前 N 个 token 再追加一遍」的重复问题。自动重连在这里的价值仅是「恢复连接」，恢复后由用户决定是否重试，而不是把半截答案悄悄续完。

---

## 5. 多轮对话集成

### 5.1 上下文管理：会话内存态 + 最近 N 轮 + 字符预算截断

- **会话存储**：进程内 `dict[conversation_id] = Conversation`，`Conversation` 持有一个 `messages: list[{role, content}]`。MVP 单用户、本地进程，**不做跨重启持久化**（重启即清空，可接受；v0.2 视需求落 SQLite）。
- **截断策略（对应验收 8「最近 N 轮，N 可配置」）**：
  - 保留最近 **N 轮问答（默认 10，配置项 `chat.history_turns`）**，超出的最老轮次丢弃；
  - 同时用**字符预算**兜底（默认 `chat.history_max_chars ≈ 4000`，按「中文字符 ≈ 1 token 略保守」的启发式近似 token 预算；本地无法低成本拿 tokenizer 计数，MVP 用字符近似，v0.2 再评估精确计数），超出时从最老轮次丢弃；
  - 截断发生时，在 `meta` 事件里带 `history_truncated: true`，前端提示「较早的对话已被截断」，满足验收 8 的「并提示用户」。
- **组装顺序**：`[system(角色/来源标注指令)] + [最近历史 user/assistant 交替] + [本轮 user]`。
- **流式答案并入历史**：服务端生成器在产出 `token` 帧的同时，在内存里 `full += delta` 累积；`done` 时把 `{"role":"assistant","content":full}` push 进 `Conversation.messages`。**不依赖前端回传**，因此即使用户中途关闭页面，已生成的完整答案仍计入历史，避免下一轮引用「上一轮答案」时空洞。前端展示的答案只是视图，权威历史在后端会话对象。

### 5.2 检索与多轮的关系

- **推荐（MVP）**：检索查询 = **当前问题**（整段嵌入，对齐 `04-技术架构.md` §6.1）；多轮历史**只进生成 prompt**（作为上下文），不参与向量检索。
- **追问退化的轻量兜底**：当当前问题很短（如「然后呢？」「那个方案后来怎样」，启发式：去空白后字符数 < 8），用户大概率在指代上一轮内容。此时用**上一轮 assistant 答案（或上一轮 user 问题）**作为检索查询兜底，一次替代查询即可。低成本、显著改善追问命中。
- **备选（v0.2）**：历史感知查询改写（LLM 把「当前问题 + 历史」改写为独立可检索问题）。MVP 不引入，因增加一次模型调用与延迟。
- **说明**：`03-功能需求.md` 验收 7 表述为「历史轮次的问答摘要或相关上下文带入**检索与生成**」——MVP 以「带入生成 + 追问兜底进检索」的最小实现满足；「摘要压缩历史再检索」列为 v0.2 增强。

---

## 6. 错误处理与指标

### 6.1 错误通知：事件类型区分数据 vs 错误

SSE 一旦开始，HTTP 状态码固定 200，**错误只能通过事件类型表达**。分层设计：

| 错误场景 | 处置 | 前端呈现 |
|---|---|---|
| 对话模型未配置 / 配置不可达（前置校验） | 首个事件即 `error`，code=`model_not_configured`，随后关闭 | 错误条「请先在设置中配置对话模型」 |
| 检索为空 / 低置信（**非错误**） | `done`，outcome=`no_hit`，附 message（「未检索到足够相关内容」，附最相关片段供判断） | 展示无命中提示 + 最相关片段（验收 5） |
| 知识库为空 | 同上 `no_hit`，message 引导先导入/同步数据（验收 6） | 引导文案 |
| 上游模型异常（HTTP 4xx/5xx、网络错） | 已流出内容**保留**；发 `error`，code=`upstream_error`，`partial=true`，随后关闭 | 保留已渲染文本 + 错误条 + 重试（验收 9） |
| 上游超时（超过 `chat.timeout`，如 60s） | `error`，code=`upstream_timeout`，`partial=true` | 同上 |
| 内部异常 | `error`，code=`internal`；日志记录 request_id + 脱敏堆栈（不含正文/密钥，对齐 §5.3） | 通用错误条 + 重试 |

实现要点：生成器内用 `try/except` 捕获 `openai`/`httpx` 异常，转成 `error` 帧再 `return`；**不要把异常透传给 StreamingResponse 让它裸断**（否则前端只能看到 transport error，没有 code/message）。

### 6.2 可观测埋点建议

- **进程内指标**（`metrics.py`，环形缓冲，容量如最近 200 条生成）：
  - **TTFT（首 token 延迟）**：路由收到请求 → 发出第一个 `token` 帧的时间（含检索 + 模型首 token）。对齐 `05-范围与路线图.md` §8.1 验收 `首 token < 3s`（本地）、`04` §7 假设 `0.5–2s`。
  - **吐字速率 tok/s**：`total_tokens / (done - 首token) `，对齐 §7「20–60 tok/s；< 10 tok/s 视为不可用」。
  - 辅助：`total_chars`、`total_tokens`、`retrieval_ms`、`provider`（本地/云端）、`model`、`request_id`。
- **出口**：
  - `done` 事件负载自带 `ttft_ms`/`tok_per_s`/字数 → 前端可在答案尾部展示「首字 X.Xs · 38 tok/s」，也是验收的即时可视证据；
  - `GET /api/metrics`（JSON）返回环形缓冲，供开发调试；
  - 结构化日志（INFO）：仅 `request_id + code + ttft_ms + tok_per_s`，**不含问题/答案/上下文正文**，满足日志脱敏红线。
- **token 计数口径**：优先取上游 `usage.completion_tokens`（本地 Ollama 通常提供；`stream_options.include_usage` 需要端点支持，取不到时以 `chunk` 数为近似或前端按字符换算），不因缺失而失败。

---

## 7. 边界：MVP 不做什么，何时升级

### 7.1 MVP 明确不做的流式能力

| 能力 | 说明 | 何时需要 |
|---|---|---|
| 客户端中途修改正在进行的请求（改问题/插入指令/换模型即时生效） | SSE 单向，改请求只能「断开重发」 | 需要双向控制时 → WebSocket |
| 服务端主动推送（导入进度/后台任务状态实时广播到常开连接） | MVP 用**轮询**（如 `/api/sync/status` 定时拉取） | 频繁后台事件、要求实时 → 独立 SSE 通道或 WebSocket |
| WebSocket 叠加 | 双向通道的复杂度（握手/心跳/重连/反代适配）MVP 用不上 | 见上两条 + 低延迟双向交互 |
| 断线后服务端续传（resume 已生成部分） | 见 §4，成本收益不划算 | 当「长答案 + 弱网络」成为高频场景（个人本地场景概率极低） |
| 同会话多路并发流（同一会话同时两路生成） | MVP 单会话同一时刻仅一路：前端提交即禁用该会话输入，后端会话级轻量锁 | 多端同步/团队场景 |
| 流内来源穿插（token 与来源卡片交替出现） | 来源统一走 `meta` 首帧，流只跑正文 | 交互升级诉求时评估 |
| 二进制/字节级流（音频等） | SSE 面向文本；二进制可 base64 但非本产品诉求 | 无 |
| 速率控制/背压 | SSE 无背压，模型本身是速率上限；本地「停止」即可限流 | 共享机器保护资源时 |

### 7.2 升级信号（何时需要离开纯 SSE）

1. 出现 **双向低延迟交互** 需求（中途改请求、即时打断换方向、工具调用协商）→ 叠加 WebSocket 通道（`StreamOutput` 抽象已隔离）。
2. 出现 **服务端主动事件** 需求（后台任务实时广播）→ 先考虑独立只读 SSE 端点（成本远低于 WebSocket），再评估 WebSocket。
3. 本地反代/浏览器对长连接限制导致 **频繁断连** → 先调大 `retry`、加心跳、确认非 404 超时；仍频繁才考虑协议升级。
4. 多端/远程访问（v1.0 开放局域网）→ 需补鉴权，SSE 仍可用（无自定义 header 的鉴权走 query/cookie），但需重新审视超时与心跳策略。
5. **性能指标不达标时先查模型与硬件，而非协议**：首 token > 3s 或 < 10 tok/s 时，先降模型档/量化，协议不是瓶颈。

### 7.3 与既有文档的边界一致性

- 私有数据不出机：SSE 全部为本地回环流量；云端 Provider 的出网收敛在 `ChatModel` 适配层内（对齐 `04` §5.2 出网清单）。
- 前端 XSS 红线：答案/来源一律文本节点或白名单渲染器，`innerHTML` 禁用（对齐 `04` §5.4）。
- 日志脱敏：指标与错误日志不含正文/密钥（对齐 `04` §5.3、`03` §4.8 验收 6/7）。
- 验收对齐：验收 2（流式增量渲染）、5（no_hit）、6（空库引导）、7（历史带上下文）、8（N 轮可配+提示）、9（异常保留已流出+重试）均在本文件有落地机制。

---

## 8. 待确认/后续项

- `chat.history_turns`（默认 10）、`chat.history_max_chars`（默认 ~4000）、`chat.timeout`（默认 60s）等常量在 v0.1 用作者真实语料实测后校准。
- 会话是否落盘 SQLite（跨重启保留多轮）→ v0.2 决策；MVP 内存态。
- 历史压缩摘要（重排/压缩替换老历史）→ v0.2；MVP 直接截断丢弃。
- `GET /api/metrics` 是否暴露到 UI「性能」面板 → 实现期定。
