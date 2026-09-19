# 分级相关性数据集（RAG-REL-001 §7.1/§7.2/§10.1）

本目录保存 **T11 分级相关性数据集**：在真实本机知识库上标注的 C01–C06 六类问题，
用于计算 PRD 10.1 的六项相关性指标（`CandidateCoverage`、`RequiredEvidenceRecall`、
`ReturnedPrecision`、`HardNegativeAdmissionRate`、`EmptyFalseContextRate`、
`IrrelevantCharacterRatio`）。

## 隐私口径（PRD §7.2，本目录的核心约束）

**仓库里只有「结构 + 脱敏 manifest + 统计」，没有任何私人正文。**

| 内容 | 存放位置 | 是否进仓库 |
|---|---|---|
| 数据集结构（`case_id`/`group_id`/`split`/等级/角色/`needs_answer`/`scope_id`/`review_status`/计数） | `cases-C0X.json` | ✅ |
| 私文内容的 SHA-256（`query_sha256`、`case_sha256`）与统计计数 | `cases-C0X.json` | ✅（哈希不算私文） |
| 统计与口径（题数、等级/角色分布、族与集合划分、语料哈希） | `manifest.json` | ✅ |
| 问题原文、来源文件名 basename、标题 `heading`、证据片段正文（`required_evidence`/`irrelevant_spans`/`relevant_spans`）、标注 `note` | `.tmp_probe/rag-relevance/relevance-private/private-C0X.json` | ❌（`.gitignore` 覆盖 `.tmp_probe/`） |
| 候选正文、查询向量、服务原始响应 | `.tmp_probe/rag-relevance/` | ❌ |

> 因此 `cases-C0X.json` **不能单独用于评测**：必须在本机用 `loader.py` 把「结构 + 私文」
> 组装成完整数据集。三个入口都会**明确报错**（不会把缺失的来源/证据当成「不相关」，避免指标失真）：
> - `load_relevance_cases` / `load_relevance_dataset`：未组装的结构 + 未提供本地私文 → `ValueError`；
> - `validate_relevance_cases`：`allow_redacted=False` 时遇到脱敏案例 → `ValueError`；
> - **`evaluate_relevance`（指标入口）**：`case.is_redacted` 为真 → `ValueError`（文案含「数据集未组装」，
>   并列出未组装的 `case_id`）——否则会产出「`precision_macro=None` + `empty_result_rate=1.0`
>   但 `precision_valid=True`」的报告，让门禁把**数据缺失误报成 A03/A04 质量不达标**（T23-F1）。

## 文件

| 文件 | 内容 |
|---|---|
| `cases-C01.json` … `cases-C06.json` | 每类一个文件：20 题（10 校准 + 10 独立测试）的**脱敏结构** + 私文哈希 + 统计 |
| `manifest.json` | 脱敏 manifest：题数、等级/角色分布、族与集合划分、语料内容哈希、标注口径 |
| `loader.py` | 本地加载器：结构 + 私文 → 完整数据集；也是评测与测试的唯一入口 |
| `private_guard.py` | 防回归自检：断言私文里的来源名/标题/原文/片段不出现在仓库文件里（默认整仓模式，见下） |

私文目录（本机，Git 忽略）：`.tmp_probe/rag-relevance/relevance-private/`
—— `private-C01.json` … `private-C06.json`（组装所需）+ `_originals/cases-original.json`
（脱敏迁移前的原始数据整份留档）+ `sensitive-needles.json`（点名敏感串：人名/可识别时间戳）。
路径可用参数或环境变量
`EVERYTHING_RAG_RELEVANCE_PRIVATE_DIR` / `EVERYTHING_RAG_RELEVANCE_CASES_DIR` 覆盖。

### 防回归自检（T21 → T22 扩展）

```powershell
python scripts/relevance_cases/private_guard.py                 # 整仓模式（默认）
python scripts/relevance_cases/private_guard.py --dataset-dir   # 只扫本目录（最严阈值）
```

- **整仓模式** needle 来自五类本地私文：本数据集私文、语料清单
  `corpus-inventory.json`（全部来源 basename/stem + 标题 + 正文预览）、
  `benchmark-private/cases-private.json`（benchmark 用例的问题与证据串）、
  `local-eval-cases.json`（三个一次性评测脚本的用例表），外加 `sensitive-needles.json`
  里的点名敏感串与**显式登记的短片段**。并豁免「仓库自身同名文件」这类歧义 needle
  （`CLAUDE.md`/`README.md`）。
- **退出码**：`0` 干净；`1` 有未豁免的私文命中；`2` 干净但**发布阻断项**仍在。
- **发布阻断项（必须显式可见，不得静默通过）**：
  `scripts/rag_eval_cases.json`（检索黄金集 72 正例/10 负例）仍含真实来源名与标题；
  按用户决定留到校准批次处理。整仓模式把它单列进 `release_blockers` 并置
  `release_blocked=true`，CLI 打印 `[RELEASE BLOCKER]` 且退出码 2。
  对应的测试是 `backend/tests/test_relevance_datasets.py::test_private_strings_absent_from_whole_repo`。
- 反向自检：把私文片段植入数据集目录 / 仓库式目录都必须被抓到
  （`test_private_guard_detects_a_planted_leak`、`test_repo_guard_detects_a_planted_leak`）。

#### needle 阈值口径（**不允许再用「阈值挡住了」解释遗漏**）

CLI 会把每个 kind 的阈值与理由一起打印；下表与 `private_guard.MIN_CHARS_BY_KIND` /
`THRESHOLD_RATIONALE` 同源（改一处必须同时改另一处）：

| kind | min_chars | 理由 |
|---|---|---|
| `sources` / `corpus_sources` | 4 | 来源 basename 自带 `.md`，≥4 字符即精确标识；`a.md` 这类 3 字符通用短名是夹具假阳性 |
| `source_stems` / `corpus_stems` | 5 | 去扩展名后 4 字中文名（如「公司介绍」）是通用短语，会与文档正文撞词；`.md` 全名仍按 4 字算 |
| `headings` | 12 | 章节标题多是「序号 + 小标题」这类通用短语，短标题在中文文档里撞词概率高 |
| `corpus_headings` | 16 | 语料标题取更保守阈值，只保留真正独特的标题 |
| `queries` / `benchmark_queries` / `local_eval_questions` | 12 | 真实提问是整句，≥12 字即身份指纹；更短多为「什么是 adb」这类通用问法 |
| `evidence` / `benchmark_evidence` / `span_texts` | 16 | 逐字证据片段短片段极易撞词 |
| `notes` | 24 | 标注依据是自然语言，取 24 字避免与报告正文撞词 |
| `corpus_previews` | 24 | 语料清单里的正文预览 |
| `local_eval_anchors` | 5 | 一次性评测脚本的「期望命中子串」（文档名片段） |
| `sensitive` | 2 | **点名列举**的敏感串（人名/可识别时间戳），不做通用短语过滤 |
| `declared_short_fragments` | 2 | **显式登记**的短私文片段，**无阈值豁免，命中即泄漏** |

> **短于同类阈值的已知私文片段怎么处理**（T23-F4 的教训）：不得靠阈值蒙混。必须把它写进
> 本地 `sensitive-needles.json` 的 `short_fragments`（含 `why`/`was_in` 说明），由
> `declared_short_fragments` 类以最严阈值扫描；CLI 会打印该类的条数与 sha256 前缀。
> 回归防线：`backend/tests/test_relevance_datasets.py::test_declared_short_fragments_absent_from_repo`
> 会点名断言 `docs/诊断报告-检索质量与生成行为.md` 不再含该片段，
> `test_guard_documents_threshold_rationale` 断言阈值理由必须可机读。

> **数据集目录模式（`--dataset-dir`）例外**：该模式只扫本目录，所有 kind 统一用 2 字符阈值、
> 不做通用短语过滤（宁严不漏）。因此**本目录内的新增文案也要避免与 ≤4 字的语料标识完全同形**——
> T24 收口时就因为阈值说明里写了示例标题（命中语料里的 2/4 字标题）而被该模式抓到，改词后才恢复 0 命中。

## 数据格式（脱敏结构）

```jsonc
{
  "schema_version": 1,
  "redaction": {                       // 声明这是脱敏结构；缺私文评测会被拒绝
    "policy": "PRD RAG-REL-001 §7.2",
    "private_fields": ["query", "targets[].source", "…"],
    "private_dataset": ".tmp_probe/rag-relevance/relevance-private/private-C01.json",
    "loader": "scripts/relevance_cases/loader.py"
  },
  "family": "C01",
  "corpus_sha256": "…",                // 语料内容哈希；过期即拒绝加载
  "annotation_protocol": { "protocol": "…", "grading_rule": "…" },
  "counts": {                          // 统计（不含来源名/标题）
    "cases": 20, "targets": 169, "calibration": 10, "test": 10,
    "targets_with_heading": 157, "targets_with_required_evidence": 107,
    "required_evidence_items": 107, "irrelevant_span_items": 0, "relevant_span_items": 0,
    "distinct_sources": 44, "distinct_headings": 120,
    "grades": { "0": 23, "1": 95, "2": 31, "3": 20 },
    "roles": { "allowed": 30, "forbidden": 117, "required": 22 },
    "required_groups": { "required_targets": 22, "required_effective_groups": 22,
                         "declared_equivalent_groups": 0 }
  },
  "cases": [
    {
      "case_id": "C01-015",
      "family": "C01",
      "group_id": "C01-g-fenbiao",     // 同源改写 / 同证据族 / 同干扰块族 → 同组，整组只进一个集合
      "split": "calibration",          // calibration | test
      "needs_answer": true,            // false = 库里确无答案的问题（C05）
      "scope_id": "global",
      "review_status": "reviewed",
      "query_sha256": "…",             // 问题原文的 SHA-256（原文在本地私文里）
      "case_sha256": "…",              // 该题全部私文字段（query + 每个目标的来源/标题/证据/范围/note）的 SHA-256
      "targets": [
        {
          "target_id": "C01-015-T1",
          "role": "required",          // required | allowed | forbidden
          "grade": 3,                  // 0/1/2/3 或 "unjudged"
          "group": ""                  // 必要证据族；等价证据共组，指标按组计一次
        }
      ]
    }
  ]
}
```

`query_sha256` / `case_sha256` 既是脱敏冗余，也是**本地完整性校验**：私文与仓库结构必须
逐字对得上（口径见 `loader.case_digest`：目标按 `target_id` 排序，与文件内顺序无关），
否则加载器直接抛错，避免「换了私文却拿旧结构算指标」。

本地私文文件格式：`{schema_version, family, corpus_sha256, cases:[{case_id, query, note,
targets:[{target_id, source, heading, required_evidence, irrelevant_spans, relevant_spans, note}]}]}`。

等级语义（PRD 7.1）：`0` 含义不同/对象错误/完全无关；`1` 仅话题相近；
`2` 有效支持一个明确需求；`3` 直接给出关键结论/步骤/数值/必要证据。
**未标注项是 `unjudged`，不是 0**；一旦进入最终结果，该次 Precision 评测无效。

## 两级分组（PRD 7.2，必须区分）

| 层级 | 字段 | 含义 | 约束 |
|---|---|---|---|
| 案例级 | `group_id` | 同一原始问题的改写、同一关键证据族、复用的干扰分块族 | **整组只进入一个集合**（calibration 或 test 之一），跨集合即被 `validate_relevance_cases` 拒绝 |
| 目标级 | `target.group` | 证据族：真正互为等价（同一事实的不同块表述）的目标 | 必须写成 `<case_id>:<族名>`，**不得跨题、跨集合复用**；成员至少两个且来自不同块；留空表示该目标自成一族（按 `target_id` 计一次） |

> 只有真正等价才共用 `group`。把不同题的证据塞进同一个族名（例如到处写 `g1`）会被
> `validate_relevance_cases` 直接拒绝，并会被 `backend/tests/test_relevance_datasets.py`
> 的断言抓住；因此没有等价证据时 `required_group_total` 等于必要证据目标数。
> 块级等价（来源 + 标题不同）需要私人正文，只有组装后才会校验。

## 校验与复现

```powershell
# 0) 本机私文是否齐备 + 能否组装（只打印统计与路径，不打印私人正文）
python scripts/relevance_cases/loader.py --assemble
#    期望：totals={'cases': 120, 'targets': 729, …}；assembled 里 redacted_remaining=0

# 1) 仓库内契约（不需要私文，CI 可跑；需要私文的用例会 skip 并说明原因）
cd backend && <venv>/python -m pytest tests/test_relevance_datasets.py tests/test_benchmark_dataset.py tests/test_rag_quality.py -q

# 2) 防回归自检：私文里的来源名/标题/原文/片段不得出现在仓库文件里（无私文时跳过并说明）
python scripts/relevance_cases/private_guard.py                 # 整仓模式（exit 0/1/2，见上）
python scripts/relevance_cases/private_guard.py --dataset-dir   # 只扫本目录（最严阈值）

# 3) 与真实索引逐字核对（需要本机知识库 + 私文）
python .tmp_probe/rag-relevance/relevance_tools.py verify --cases scripts/relevance_cases --out .tmp_probe/rag-relevance/verify-t21.json
# errors=0 表示：来源/标题存在于真实索引、证据片段逐字命中、分组不跨集合、每类 10+10

# 4) 六指标基线（未经筛选的现状，六类一起跑；需要本机知识库 + 私文）
python .tmp_probe/rag-relevance/relevance_tools.py baseline --cases-dir scripts/relevance_cases --private-dir .tmp_probe/rag-relevance/relevance-private --out .tmp_probe/rag-relevance/relevance-baseline-all.json

# 5) benchmark 用例集（结构 + 本地私文；缺私文时 --json 里 available=false）
python benchmark/cases_loader.py
python benchmark/validate_cases.py     # 逐条核对证据串确实来自 gold 块
```

**无私文环境（CI / 公开仓库克隆）**：结构、分组、manifest、脱敏契约用例照常通过；
需要私人正文的用例（组装、逐字校验、防回归自检）`skip`，原因固定为
`private dataset not present: 本地私文 … 缺失（Git 忽略，仅本机存在）`。
既不静默通过，也不报成失败而与真实缺陷混淆。

## 生成与脱敏流程（可复现）

1. 写问题清单 `questions-C0X.json`；用 `relevance_tools.py candidates` 转储**真实候选并集**
   （dense/BM25 排名、门控与选择原因、最终 8 条来源、送入模型的有效正文）。
2. 逐块判读并写 `notes-C0X.notes`（`notes_to_labels.py` 转成 `labels-C0X.json`，引用候选序号或块引用前缀）。
3. `build_cases.py` 解析真实 source/heading/字符范围/哈希并生成完整 `cases-C0X.json`；
   任何证据片段不是真实原文都会直接报错。
4. **脱敏迁移**（T21）：`python .tmp_probe/rag-relevance/t21_redact.py --manifest`
   —— 把完整数据拆成「仓库结构 + 本地私文」，逐字段回环校验后重建 `manifest.json`。
   原始数据整份留档在私文目录的 `_originals/cases-original.json`，可重复运行（幂等）。
5. 每次改动数据集后必须重跑
   `backend/tests/test_relevance_datasets.py`（含 manifest 漂移与防回归自检）。

## 复现口径说明（重要）

- **标题来源**：本基线使用 `HybridRetriever` 的融合候选池与最终 `RetrievedChunk`，
  `heading_path` 直接来自块 metadata；线上 `evidence_view` 走 `HeadingLookup.for_vectorstore(store)`
  补齐标题。两条路径**都带标题、取值同源**（同一 `heading_path` 字段），因此不存在
  「评测不带标题、线上带标题」的口径分叉。T29 之后改用 `evidence_view` 的脚本必须显式传
  `heading_lookup=HeadingLookup.for_vectorstore(store)`。
- **scored / returned 定义**：`scored` = dense top-40 ∪ BM25 top-40 ∪ 最终返回集
  （hard negative 的分母必须包含被放行的块，否则放行率会大于 1）；`returned` = 生产最终 8 条来源。
- **`window_candidates` / `select_from_scores` 的前置条件（T19-F9，调用方责任）**：
  `select_from_scores` 接受的候选必须是**已经过** `relevance_selector.window_candidates`
  的窗口结果，**且**该次窗口截取里 `outside` 候选的 `outside_candidate_window` 记录必须由调用方
  通过 `exclusions` 一并传入；本函数不重复做窗口截取，否则截取口径会与线上分叉
  （不变量：截取前全部候选 == `selected ∪ exclusions`，其中 `-2` 的「缺分」不新增原因类别）。
  等价地，若要自建第三条调用路径（例如既筛选又写来源字段），必须复用同一条窗口 + 筛选链路，
  不得只把子集喂给 `select_from_scores`。
- **ReturnedPrecision**：宏平均只统计「有返回结果且全部视图已标注」的问题；空结果的 Precision
  记为未定义并单独报告空结果率；出现 unjudged 视图时该次 `precision_valid=false`。
- **IrrelevantCharacterRatio 计数方法**：分子为最终正文中**人工标注的无关原文范围**字符数，
  按唯一范围（`case_id + block_ref + 片段`）去重只计一次；等级 0/1 的整块按整段计入，
  等级 2/3 的块只计显式标注的 `irrelevant_spans`；分母为最终正文字符总数。
- **运行间抖动**：查询嵌入服务非确定性（同文本两次嵌入最大绝对差约 4e-3），
  逐题返回集可能相差 1 条，导致 recall/precision 在 ±1 题内抖动；跨批次比较请用同一口径重跑。

## 基线六指标（未筛选现状，120 题）

下表的数值在**脱敏前**（2026-09-19）测得；T21 脱敏后用「结构 + 本地私文」重新组装并**完整复跑**
六类 120 题 / 729 目标，指标复现到同一口径（整体 `CandidateCoverage 1.0` /
`RequiredEvidenceRecall 0.95082` / 空结果率 0.05 / `EmptyFalseContextRate 0.75` /
`precision_valid true`，C01–C06 逐类同量级；逐题差异来自嵌入服务非确定性 ±1 题）。

| 类别 | CandidateCoverage | RequiredEvidenceRecall | ReturnedPrecision(宏) | ReturnedPrecision(视图) | HardNegativeAdmissionRate | IrrelevantCharacterRatio | 空结果率 | EmptyFalseContextRate | precision_valid |
|---|---|---|---|---|---|---|---|---|---|
| C01 同词异义 | 1.0 | 0.9091 | 0.3650 | 0.3704 | 0.4000 | 0.6230 | 0.05 | — | true |
| C02 同领域不同提问目标 | 1.0 | 1.0 | 0.3179 | 0.2377 | 0.4732 | 0.7643 | 0.0 | — | true |
| C03 多主题分块 | 1.0 | 1.0 | 0.4029 | 0.3500 | 0.4952 | **0.5724** | 0.0 | — | true |
| C04 综合问题与局部证据 | 1.0 | 0.9000 | 0.5119 | 0.4836 | 0.5145 | 0.4866 | 0.0 | — | true |
| C05 近似内容资料不足 | 不适用 | 不适用 | 0.0 | 0.0 | 0.4609 | 1.0000 | 0.25 | **0.75（15/20）** | true |
| C06 补充内容引入无关内容 | 1.0 | 1.0 | 0.4000 | 0.2273 | 0.4545 | **0.7594** | 0.0 | — | true |
| **整体** | **1.0** | **0.9508** | **0.3473** | **0.3097** | **0.4624** | **0.6548** | **0.05** | **0.75（15/20）** | **true** |

**A06 量化起点**（PRD 10.2 A06：C03、C06 的无关字符比例比基线降低 ≥50%）：
- C03 基线 `0.5724` → 目标 **≤ 0.2862**
- C06 基线 `0.7594` → 目标 **≤ 0.3797**

**A05 量化起点**（C05 无证据问题误放行率为 0）：C05 基线 `EmptyFalseContextRate = 0.75`
（分母 20 道 `needs_answer=false` 题，其中 15 道返回了知识上下文）。

## 现状

- 六类数据集齐备：`cases-C01.json` … `cases-C06.json`，每类 20 题（10 校准 + 10 独立测试），
  共 120 题 / 729 个标注目标；仓库内为脱敏结构，本机私文齐备时可完整组装并通过校验。
- 基线产物：`.tmp_probe/rag-relevance/relevance-baseline-all.json`（逐题 trace）与
  `.tmp_probe/rag-relevance/relevance-baseline-summary.json`（六类对照表与 A06 目标值）。

## 发布阻断项（必须清零才能发布）

| 项 | 内容 | 处理 |
|---|---|---|
| `scripts/rag_eval_cases.json` | 检索黄金集（72 正例/10 负例）仍含**真实来源名与标题** | 按用户决定留到**校准批次**（T30/T32）处理；在此之前整仓自检退出码为 2 并打印 `[RELEASE BLOCKER]`，不得当作「已脱敏」 |

T22 已清理的同类载体（供对照）：`benchmark/*.json` 数据产物（移入
`.tmp_probe/rag-relevance/benchmark-private/`，`.gitignore` 忽略；`cases.json` 改为脱敏结构 +
`benchmark/cases_loader.py` 组装）、`backend/tests/**` 内联的真实来源名/标题/问题、
`scripts/eval_retrieval*.py` 内联用例表（移入 `local-eval-cases.json` + `scripts/local_eval_cases.py`）、
`docs/诊断报告-检索质量与生成行为.md` 与 `benchmark/BENCHMARK_REPORT.md` 中的文档名（改为 `doc-noise-NN.md`）。
