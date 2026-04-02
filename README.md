# ComfyUI Easy RAG (Local)

本插件为 ComfyUI 提供本地 RAG 节点，能力类似 LM Studio `rag-v1`：

- 文档上传/加载（`txt`, `json`, `md`, `pdf`）
- 自动文本切分 + `sentence-transformers` embedding
- 使用 `FAISS` 构建向量数据库
- 查询执行 `top-k` 语义检索
- 自动拼接 context 到 prompt
- 通过 LM Studio 本地 OpenAI 兼容 API 生成回答
- 预制库支持（快速加载预设知识库）
- 两阶段 RAG 检索（专为 Z-Image 提示词优化）
- 文档角色智能路由检索

## 节点列表

1. `EasyRAG - 文档加载`
2. `EasyRAG-预制库`
3. `EasyRAG - 向量库构建(FAISS)`
4. `EasyRAG - LM Studio API (高级)`
5. `EasyRAG - LM Studio API (简约)`
6. `EasyRag-Zimage-2步检索推理`

## 语言设置

在 ComfyUI 的设置里可以找到 `EasyRAG 语言`：

- `跟随 ComfyUI`
- `中文`
- `英文`

切换后刷新页面即可生效。后端运行提示和节点面板文案都会跟着切换。

## 安装

在当前插件目录安装依赖：

```bash
pip install -r requirements.txt
```

将本目录放入：

```text
ComfyUI/custom_nodes/comfyui-easy-rag
```

重启 ComfyUI。

## 工作流示例

### 方式 A：分步 RAG

1. 文档加载节点：
   - `document`：选择 `input` 目录中的文档（仅支持 `txt/json/md/pdf`）
   - `上传文档` 按钮：直接弹出文件选择框上传文档到 `ComfyUI/input`
2. 向量库构建节点接收 `documents`，生成 `rag_index`
   - `chunk_size` 默认 `400`
   - `chunk_overlap` 默认 `80`
   - `show_retrieval_log` 可开启检索日志，后续问答时自动输出命中与分数
3. LM Studio API（高级）节点输入 `question`，并连接 `rag_index` 生成回答（无 `rag_context` 输入）
   - 可选连接 `image` 输入，走多模态问答
   - `model` 下拉会自动获取 LM Studio 可用模型列表
   - 统一使用 OpenAI 兼容接口，默认走 `/v1/responses`，不可用时自动回退 `/v1/chat/completions`
   - 输出内容字段自动适配（`content` / `reasoning_content`）
   - `stream` 支持流式返回并在控制台实时输出增量文本
   - `unload_model_after_response` 可在回答后请求卸载模型以节省显存
   - `clear_vram_before_run` 运行前清理显存
4. LM Studio API（简约）节点
   - 输入更少：去掉 `temperature` / `max_tokens` / `top_k` / `stream`
   - 输出只有 `answer`

### 方式 B：一体化问答

1. 文档加载 -> 向量库构建
2. 将 `rag_index` 直接连接到 `LM Studio API (高级)` 或 `LM Studio API (简约)` 节点
3. 节点会自动执行检索并把 context 拼接到 prompt 后请求 LM Studio

### 方式 C：预制库加载

1. 使用 `EasyRAG-预制库` 节点直接加载预设知识库
   - 选择 `rag/` 目录下的预制库文件夹
   - 预制库已内置 Z-Image 提示词工程知识库（Z-Image_1steps、Z-Image_2steps）等

### 方式 D：两阶段 RAG（Z-Image 提示词生成）

1. 加载预制库或自建向量库（包含范式、词汇、范例文档）
2. 使用 `EasyRag-Zimage-2步检索推理` 节点
   - **第一阶段**：从范式文档中检索，选择合适的提示词结构，生成 7 层骨架
   - **第二阶段**：根据骨架从词汇库和范例库中检索，填充并生成最终提示词
   - 支持自定义 stage1 和 stage2 的系统提示词
   - 自动按文档角色路由检索（paradigm/vocabulary/example）
   - 专门为 Z-Image（万相）模型中文写实人像提示词优化

## 文档角色智能路由

插件支持根据文件名自动推断文档角色，用于精准检索：

- `paradigm`：范式文档（文件名包含 "paradigm"）
- `vocabulary`：词汇文档（文件名包含 "vocabulary" 或 "vocab"）
- `example`：范例文档（文件名包含 "example"）
- `system`：系统提示文档（文件名包含 "system_prompt" 或 "system-prompt"）
- `general`：通用文档（默认）

两阶段检索时会自动按角色过滤：
- Stage1：检索 paradigm/system/general 文档
- Stage2：检索 vocabulary/example 文档

## LM Studio 配置

确保 LM Studio 已启动本地服务（OpenAI 兼容 API），通常地址：

```text
http://127.0.0.1:1234
```

如果 `model` 留空，节点会自动从 `/v1/models` 读取第一个可用模型。

当前实现同时支持：

1. OpenAI 兼容推理接口：`/v1/chat/completions`
2. OpenAI 兼容 Responses 接口：`/v1/responses`
3. Native v1 模型管理接口：
   - `GET /api/v1/models`（模型列表）
   - `POST /api/v1/models/unload`（卸载模型）

## 向量模型放置位置（推荐）

本插件只从 `ComfyUI/models/embeddings` 读取本地 `sentence-transformers` 模型目录，不支持其他路径或在线模型名。

示例目录：

```text
ComfyUI/models/embeddings/
  all-MiniLM-L6-v2/
    config.json
    modules.json
    ...
  bge-small-zh-v1.5/
    config.json
    modules.json
    ...
  Qwen3-Embedding-2B/
    config.json
    ...
```

`向量库构建` 节点中的 `embedding_model` 下拉会自动列出这些本地目录。
如果目录为空，节点会提示你先放入本地 embedding 模型。

推荐模型（按常见用途）：

1. `BAAI/bge-small-zh-v1.5`：中文语义检索，速度快，显存/内存占用低
2. `BAAI/bge-base-zh-v1.5`：中文效果更好，资源占用高于 small
3. `sentence-transformers/all-MiniLM-L6-v2`：英文与通用场景，轻量稳定
4. `intfloat/multilingual-e5-small`：多语言场景，体积较小
5. `intfloat/multilingual-e5-base`：多语言效果更好，资源占用更高
6. `Qwen/Qwen3-Embedding-2B`：支持 instruction-aware 的 Qwen3 系列，两阶段检索效果最佳

## 向量库存储位置

默认保存在：

```text
data/faiss_indexes/<index_name>/
```

包含：

- `index.faiss`
- `chunks.json`
- `meta.json`
