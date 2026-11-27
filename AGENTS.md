# nooshare-rag AGENTS.md

> **通用检索引擎仓库**
>
> 本仓库是 NooShare Matrix 的公共检索基础设施，提供 RAG 核心能力、向量数据模型与检索接口。
> 不包含任何业务逻辑、业务数据或内部运行配置。

---

## 职责边界

**本仓库拥有：**

- RAG 核心检索引擎代码
- PostgreSQL / pgvector 数据模型与迁移脚本
- Embedding、Hybrid Search、Reranker 接口定义
- 通用测试框架与合成测试语料
- 公开 API 文档（仅限接口契约）

**本仓库不拥有：**

- 内部主机地址、数据库连接配置、运行时凭据 → Workenv
- 真实业务语料与数据集 → 按数据授权独立管理
- 业务专用 Prompt、领域规则、研究逻辑 → 业务仓库
- 系统级设计决策与正式设计文档 → nooshare-design
- 内部服务拓扑、网络配置、部署细节 → Workenv

---

## 公开边界

| 内容 | 可见性 | 归属 |
|------|--------|------|
| RAG 核心代码、检索接口 | Public | 本仓库 |
| PostgreSQL / pgvector 数据模型 | Public | 本仓库 |
| Embedding、Hybrid Search、Reranker 接口 | Public | 本仓库 |
| 通用测试和合成语料 | Public | 本仓库 |
| 内部主机、数据库地址、连接配置 | Private | Workenv |
| 真实业务 Corpus 和数据集 | 按数据授权管理 | 数据源仓库 |
| 业务专用 Prompt、规则 | Private | 业务仓库 |
| 系统级设计决策 | Private | nooshare-design |

**原则**：Public repo 不代表所有运行配置也公开。GitHub Actions 可以公开工作流定义，但凭据必须留在 Secrets，不得将内部基础设施信息输出到公开日志。

---

## 结构纪律

| # | 原则 | 规则 |
|---|------|------|
| 1 | **Git 是唯一记录** | 禁止产生 decisions/、history/、旁置台账、验收报告等过程产物。所有变更通过 Git 历史追溯。 |
| 2 | **平铺结构** | 禁止按认知类型预建子目录（如 notes/、references/）。同一 subdomain 下文件平铺。 |
| 3 | **按需建目录** | 禁止为假想规模预建结构。有真实资产时才建目录。 |
| 4 | **AGENTS.md 职责** | 只放规则与路由。不堆砌知识正文、过程记录或初始化状态。 |
| 5 | **文件即状态** | 文件存在即为可用，删除即为废弃。无需 status 字段或状态标记。 |
| 6 | **公共仓库文档禁令** | 禁止创建 docs/、deliverables/ 等文档目录，禁止生成或提交文字材料。README.md 不做入口（本仓库无 README）。 |

---

## 操作约束

| # | 原则 | 规则 |
|---|------|------|
| 1 | **凭据隔离** | 不把密钥、令牌、密码写入普通配置文件或文档。所有凭据通过环境变量或 Secrets 管理。 |
| 2 | **配置与代码分离** | 运行配置（数据库地址、API 端点）不得硬编码。使用环境变量或配置文件（配置文件不得提交）。 |
| 3 | **测试数据边界** | 测试语料必须是合成数据或公开数据集。禁止提交真实业务数据。 |
| 4 | **CI 日志安全** | GitHub Actions 工作流不得在日志中输出内部基础设施信息（主机名、IP、连接串）。 |
| 5 | **接口契约优先** | 公开 API 必须有明确的输入输出契约。内部实现细节不暴露。 |

---

## 写作约束

- 简短、直给、不啰嗦
- 同一信息只说一次
- 用具体动作和对象表达，避免抽象名词链
- 有依据的陈述直接说明，无依据的标注假设
- 配置优先可执行，文档只补「为什么」

---

## MVP 实现状态（2026-11-28）

**已完成的核心能力：**

1. **文档输入协议**（document.py）
   - Document 和 DocumentMetadata 数据类
   - 支持年报/半年报等文档类型
   - 包含 PIT 时间戳（available_at）

2. **文本切片**（chunker.py）
   - SectionChunker 实现章节感知切片
   - 支持中文标题模式匹配（节/章/部、一/二/三、(一)/(二)等）
   - 保留章节路径和字符偏移

3. **PostgreSQL 持久化**（migrations.py + storage.py）
   - documents 表：文档元数据
   - chunks 表：切片内容 + pgvector 向量
   - HNSW 索引：向量相似度检索
   - 全文检索索引：GIN 索引（simple 配置）

4. **检索接口**（storage.py）
   - full_text_search：全文检索
   - vector_search：向量相似度检索
   - hybrid_search：RRF 混合检索
   - 所有接口支持 available_at 过滤

5. **Embedding 接口**（embedding.py）
   - EmbeddingModel 协议定义
   - MockEmbeddingModel 测试实现
   - 真实模型接入点（OpenAI、BGE 等）

6. **测试套件**
   - test_chunker.py：切片功能测试
   - test_storage.py：存储层集成测试
   - test_retrieval_quality.py：检索质量测试框架

**已知限制：**

- 中文全文检索：PostgreSQL simple 配置分词效果差，需要安装 zhparser 或 pg_jieba
- Mock embeddings：无语义信息，检索质量测试无法验证真实效果
- 无 reranker：当前仅使用 embedding 相似度，未实现 cross-encoder 重排

**下一步：**

1. 接入真实 embedding 模型（OpenAI ada-002 或本地 BGE）
2. 安装中文分词扩展（zhparser 或 pg_jieba）
3. 实现 reranker 接口
4. 接入 nooshare-annual-semi-research 验证真实年报检索
