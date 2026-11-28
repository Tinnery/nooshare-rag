# 小模型限制与已知问题

## 概述

本文档记录 nooshare-rag MVP 中使用小模型（qwen3-embedding:0.6b）的已知限制和问题。

## 1. Embedding 模型限制

### 1.1 数字精确匹配失败

**问题**：0.6B 参数的 embedding 模型无法精确匹配数字。

**表现**：
- 查询「平安银行2024年净利润」时，包含「445.08」的 chunk 向量相似度仅 0.61-0.66
- 查询「贵州茅台2023年营业收入」时，包含「147,693,604,994.14」的 chunk 无法通过向量检索找到

**原因**：
- 小模型的 embedding 空间对数字的编码不够精细
- 数字在文本中的语义权重被稀释
- 缺乏对财务数字的专门训练

**缓解方案**：
- 依赖 FTS（全文检索）进行数字匹配（已采用，FTS 权重 0.7）
- **升级到 qwen3-embedding:4b 或 8b**（见下文模型对比）
- 在 chunk 中显式标注关键数字（待实现）

### 1.2 语义理解有限

**问题**：小模型对复杂财务概念的理解有限。

**表现**：
- 「每股收益」与「EPS」的语义关联较弱
- 无法准确理解「分红」「股利」「派息」的同义关系

**缓解方案**：
- 使用同义词扩展查询
- 依赖 FTS 的精确匹配
- 升级到更大的 embedding 模型

## 2. Embedding 模型选型对比

### qwen3-embedding 系列（推荐）

既然已在用 qwen3-embedding:0.6b，升级到同系列更大模型是最自然的选择：

| 模型 | 大小 | 维度 | 上下文 | MTEB 多语言 | 中文效果 |
|------|------|------|--------|-------------|----------|
| qwen3-embedding:0.6b | 639 MB | 1024 | 32K | 64.33 | 优秀 |
| **qwen3-embedding:4b** | **2.5 GB** | **2560** | **40K** | **69.45** | **顶级** |
| qwen3-embedding:8b | 4.7 GB | 4096 | 40K | 70.58 | 顶级 |

**推荐**：qwen3-embedding:4b（性价比最高）
- MTEB 分数从 64.33 提升到 69.45（+5.12）
- 维度从 1024 提升到 2560（更精细的语义编码）
- 大小仅 2.5 GB，RTX 4060 Ti 16GB 完全够用
- 支持 MRL（Matryoshka Representation Learning），可灵活截断维度

**升级步骤**：

    # 拉取 4b 模型
    ollama pull qwen3-embedding:4b
    # 修改 embedding.py 中的模型配置
    # self.model_name = "qwen3-embedding:4b"
    # 重新生成所有 embeddings（维度变了）

**注意事项**：
- 维度从 1024 变为 2560，需要重新生成所有 embeddings
- pgvector 的 vector 列维度需要修改（ALTER TABLE ... ALTER COLUMN ... TYPE vector(2560)）
- 4b 模型需要约 3-4 GB 显存，推理速度比 0.6b 慢

### 其他备选模型

| 模型 | 大小 | 维度 | MTEB 多语言 | 中文效果 | 备注 |
|------|------|------|-------------|----------|------|
| bge-m3 | 1.2 GB | 1024 | - | 优秀 | 支持稀疏+稠密+多向量检索 |
| mxbai-embed-large | 670 MB | 1024 | 64.68 | 一般 | 仅英文，不适合中文场景 |
| nomic-embed-text | 274 MB | 768 | 62.28 | 差 | 仅英文，适合资源受限场景 |

**结论**：qwen3-embedding:4b 是最佳选择（中文顶级 + 性价比最高）。

## 3. FTS 分词问题

### 3.1 jieba 分词与公司名缺失

**问题**：chunk 来自财务报表部分，可能不包含公司名称。

**已解决**：
- 在 chunk 预处理时添加公司名/股票代码前缀（chunker.py _build_metadata_prefix）

### 3.2 数字格式不匹配

**问题**：年报中的数字格式多样，查询时难以预测。

**缓解方案**：
- 在 ground truth 中使用年报中的实际格式
- 考虑在索引时标准化数字格式（待实现）

## 4. 检索策略

### 4.1 Hybrid Search 权重调整

**当前配置**：
- FTS 权重：0.7
- Vector 权重：0.3
- RRF k=60

**升级后调整建议**：
- 升级到 4b/8b 后，可降低 FTS 权重到 0.5-0.6
- 更大模型的语义理解能力更强，vector 检索的权重可以提高

### 4.2 Chunk 大小优化

**当前配置**：
- max_chunk_size=2000
- overlap=300

## 5. 测试用例修正记录

### 5.1 Test 3: 贵州茅台2023年营业收入

**原始 ground truth**：1,476.94
**修正后**：147,693,604,994.14

### 5.2 Test 4: 贵州茅台2023年基本每股收益

**问题**：FTS 返回 0 结果
**原因**：jieba 分词 AND 逻辑 + chunk 不含公司名
**状态**：已修复（chunk 添加公司名前缀）

## 6. 后续改进方向

1. **升级 embedding 模型**：升级到 qwen3-embedding:4b（推荐）或 8b
2. **数字标准化**：在索引时统一数字格式（待评估优先级）
3. **同义词扩展**：在查询时扩展同义词（待评估优先级）
4. **混合检索优化**：探索更复杂的检索策略（如先 FTS 后 vector rerank）

## 参考资料

- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- Qwen3-Embedding 官方文档: https://github.com/QwenLM/Qwen3-Embedding
- RRF (Reciprocal Rank Fusion): https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
