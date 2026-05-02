from __future__ import annotations

# ----------------------------
# sentence-transformers 兼容补丁 (智能自适应)
# ----------------------------
def patch_pooling():
    try:
        from sentence_transformers.models import Pooling
        import inspect
        original_init = Pooling.__init__
        
        # 避免重复打补丁
        if getattr(original_init, "_is_patched", False):
            return
            
        sig = inspect.signature(original_init)
        
        def fixed_init(self, *args, **kwargs):
            params = list(sig.parameters.keys())
            if len(args) == 0 and len(params) > 1:
                first_param = params[1]
                if first_param not in kwargs:
                    kwargs[first_param] = 768
            return original_init(self, *args, **kwargs)
            
        fixed_init._is_patched = True
        Pooling.__init__ = fixed_init
    except Exception:
        pass

patch_pooling()

# ============================
# 纯 TXT 优化版，无多余 JSON 干扰
# ============================

import hashlib
import json
import re
import threading
import io
import contextlib
import logging
import gc
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import requests
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

try:
    from transformers.utils import logging as hf_logging  # type: ignore
except Exception:
    hf_logging = None

try:
    import torch
except Exception:
    torch = None

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None

from .i18n import t


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}

# ==============================
# ✅ 【新增】嵌入编码 batch_size 默认值
#    降低此值可减少显存峰值占用，提升稳定性
# ==============================
DEFAULT_EMBED_BATCH_SIZE = 16


def _faiss_temp_file() -> Path:
    return Path(tempfile.gettempdir()) / f"easyrag_{uuid.uuid4().hex}.faiss"


def _faiss_write_index_safe(index: Any, target_path: Path) -> None:
    """Write FAISS index through an ASCII temp path for Windows Unicode-path compatibility."""
    tmp = _faiss_temp_file()
    try:
        faiss.write_index(index, str(tmp))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(tmp.read_bytes())
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _faiss_read_index_safe(source_path: Path) -> Any:
    """Read FAISS index through an ASCII temp path for Windows Unicode-path compatibility."""
    tmp = _faiss_temp_file()
    try:
        tmp.write_bytes(source_path.read_bytes())
        return faiss.read_index(str(tmp))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _safe_read_text(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding, errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        from PyPDF2 import PdfReader
    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


# ==============================
# ✅ 【新增】文件哈希计算（用于增量索引）
#    同时对文件内容+修改时间做哈希，任何变化都能检测到
# ==============================
def _compute_file_hash(path: Path) -> str:
    """计算文件的 MD5 哈希，用于检测文件变化。"""
    try:
        mtime = str(path.stat().st_mtime)
        content = path.read_bytes()
        h = hashlib.md5(mtime.encode() + content).hexdigest()
        return h
    except Exception:
        return ""


# ==============================
# 简化 JSON 解析，不干预纯文本
# ==============================
def parse_json_to_text(raw: str) -> str:
    try:
        obj = json.loads(raw)
        parts = []
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    t = item.get("text", "") or item.get("optimized_prompt", "")
                    if t.strip():
                        parts.append(t.strip())
        elif isinstance(obj, dict):
            t = obj.get("text", "") or obj.get("optimized_prompt", "")
            if t.strip():
                parts.append(t.strip())
        if parts:
            return "\n".join(parts)
    except Exception:
        pass
    return raw.strip()


def load_single_document(path: Path, encoding: str = "utf-8") -> Dict:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file: {path.suffix}")

    if ext == ".pdf":
        text = _read_pdf(path)
    else:
        raw = _safe_read_text(path, encoding=encoding)
        if ext == ".json":
            text = parse_json_to_text(raw)
        else:
            text = raw

    return {
        "source": str(path),
        "extension": ext,
        "text": text.strip(),
        "title": path.name,
    }


def expand_paths(path_text: str) -> List[Path]:
    if not path_text.strip():
        return []
    parts = re.split(r"[\n,;]+", path_text.strip())
    files: List[Path] = []
    for p in parts:
        raw = p.strip().strip('"').strip("'")
        if not raw:
            continue
        path = Path(raw)
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                files.extend(path.glob(f"**/*{ext}"))
            continue
        for hit in Path(".").glob(raw):
            if hit.is_file() and hit.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(hit.resolve())
    seen = set()
    out = []
    for f in files:
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            out.append(f.resolve())
    return out


def split_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 0) -> List[str]:
    text = re.sub(r"\r\n?", "\n", text or "").strip()
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines


# ==============================
# ✅ 【新增】嵌入前腾出 ComfyUI 显存
#    避免 ComfyUI 主模型和 embedding 模型同时占用显存导致 OOM
# ==============================
def _offload_comfyui_models() -> None:
    """
    在加载 embedding 模型前，要求 ComfyUI 将当前主模型卸载到 CPU 或内存，
    腾出 GPU 显存供 embedding 模型使用。
    """
    if model_management is None:
        return
    try:
        # 优先使用 unload_all_models（最彻底）
        if hasattr(model_management, "unload_all_models"):
            model_management.unload_all_models()
        elif hasattr(model_management, "cleanup_models"):
            try:
                model_management.cleanup_models()
            except TypeError:
                model_management.cleanup_models(True)
        if hasattr(model_management, "soft_empty_cache"):
            model_management.soft_empty_cache()
        elif hasattr(model_management, "empty_cache"):
            model_management.empty_cache()
    except Exception:
        pass

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    gc.collect()


# ==============================
# ✅ 【修改】EmbeddingBackend
#    1. 加载时使用 fp16 → 显存减半（~8GB vs ~16GB）
#    2. encode 支持 batch_size 参数 → 控制单次显存峰值
#    3. 加载前可选触发 ComfyUI 模型卸载
# ==============================
@dataclass
class EmbeddingBackend:
    model_name: str
    device: Optional[str] = None
    # ✅ 【新增】是否在加载前先卸载 ComfyUI 主模型腾显存，默认开启
    offload_comfyui: bool = True
    _model: Optional[SentenceTransformer] = None
    _MODEL_CACHE: ClassVar[Optional[Dict[str, Any]]] = None
    _MODEL_CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers 未安装")
        if EmbeddingBackend._MODEL_CACHE is None:
            EmbeddingBackend._MODEL_CACHE = {}
        if self._model is None:
            key = str(self.model_name).strip()
            cache_key = key if not self.device else f"{key}@@{self.device}"
            with EmbeddingBackend._MODEL_CACHE_LOCK:
                cached = EmbeddingBackend._MODEL_CACHE.get(cache_key)
                if cached is None:
                    # ✅ 【新增】加载前先腾出 ComfyUI 显存
                    if self.offload_comfyui:
                        print(f"[EasyRAG] 🔄 卸载 ComfyUI 主模型，腾出显存...")
                        _offload_comfyui_models()

                    out_buf = io.StringIO()
                    err_buf = io.StringIO()
                    st_logger = logging.getLogger("sentence_transformers")
                    tf_logger = logging.getLogger("transformers")
                    tfmu_logger = logging.getLogger("transformers.modeling_utils")
                    old_st_level = st_logger.level
                    old_tf_level = tf_logger.level
                    old_tfmu_level = tfmu_logger.level
                    old_hf_verbosity = None
                    try:
                        st_logger.setLevel(logging.ERROR)
                        tf_logger.setLevel(logging.ERROR)
                        tfmu_logger.setLevel(logging.ERROR)
                        if hf_logging is not None:
                            try:
                                old_hf_verbosity = hf_logging.get_verbosity()
                            except Exception:
                                old_hf_verbosity = None
                            hf_logging.set_verbosity_error()

                        # ✅ 【新增】fp16 加载，显存占用从 ~16GB 降至 ~8GB
                        model_kwargs: Dict[str, Any] = {}
                        if torch is not None:
                            model_kwargs["torch_dtype"] = torch.float16

                        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                            try:
                                if self.device:
                                    cached = SentenceTransformer(
                                        key,
                                        device=self.device,
                                        model_kwargs=model_kwargs,
                                    )
                                else:
                                    cached = SentenceTransformer(
                                        key,
                                        model_kwargs=model_kwargs,
                                    )
                            except TypeError:
                                # ✅ 兼容旧版 sentence-transformers（不支持 model_kwargs）
                                print("[EasyRAG] ⚠️ 当前 sentence-transformers 版本不支持 model_kwargs，以 fp32 加载。"
                                      "显存占用会更高，建议 pip install -U sentence-transformers")
                                if self.device:
                                    cached = SentenceTransformer(key, device=self.device)
                                else:
                                    cached = SentenceTransformer(key)
                    except Exception:
                        raise
                    finally:
                        st_logger.setLevel(old_st_level)
                        tf_logger.setLevel(old_tf_level)
                        tfmu_logger.setLevel(old_tfmu_level)
                        if hf_logging is not None and old_hf_verbosity is not None:
                            try:
                                hf_logging.set_verbosity(old_hf_verbosity)
                            except Exception:
                                pass
                    EmbeddingBackend._MODEL_CACHE[cache_key] = cached
                self._model = cached
        return self._model

    # ✅ 【修改】encode 支持 batch_size，默认 16（原来是模型默认 32）
    #    batch_size 越小，单次峰值显存越低，但整体速度略慢
    def encode(self, texts: List[str], batch_size: int = DEFAULT_EMBED_BATCH_SIZE) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,  # 大批量时显示进度
            batch_size=batch_size,
        )
        return vectors.astype(np.float32)

    def release(self):
        if self._model is not None:
            try:
                if hasattr(self._model, "cpu"):
                    self._model.cpu()
                if hasattr(self._model, "to"):
                    self._model.to("cpu")
                del self._model
                self._model = None
            except Exception:
                pass
        gc.collect()


# ==============================
# 彻底修复显存泄露：全流程强制清理
# ==============================
def unload_embedding_model(model_name: Optional[str] = None) -> Dict:
    unloaded: List[str] = []
    models_to_release: List[Any] = []
    errors: List[str] = []
    with EmbeddingBackend._MODEL_CACHE_LOCK:
        cache = EmbeddingBackend._MODEL_CACHE or {}
        if model_name is None:
            unloaded = list(cache.keys())
            models_to_release = list(cache.values())
            cache.clear()
        else:
            key = str(model_name).strip()
            remove_keys = [k for k in list(cache.keys()) if k == key or k.startswith(f"{key}@@")]
            for rk in remove_keys:
                model_obj = cache.pop(rk, None)
                if model_obj is not None:
                    models_to_release.append(model_obj)
                unloaded.append(rk)
        EmbeddingBackend._MODEL_CACHE = cache

    for model_obj in models_to_release:
        try:
            if hasattr(model_obj, "cpu"):
                model_obj.cpu()
        except Exception as e:
            errors.append(f"model.cpu failed: {e}")
        try:
            if hasattr(model_obj, "to"):
                model_obj.to("cpu")
        except Exception as e:
            errors.append(f"model.to(cpu) failed: {e}")
        del model_obj

    models_to_release.clear()

    gc.collect()
    gc.collect()

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    if model_management is not None:
        try:
            if hasattr(model_management, "cleanup_models"):
                try:
                    model_management.cleanup_models()
                except TypeError:
                    model_management.cleanup_models(True)
            if hasattr(model_management, "soft_empty_cache"):
                model_management.soft_empty_cache()
            elif hasattr(model_management, "empty_cache"):
                model_management.empty_cache()
        except Exception:
            pass

    gc.collect()

    return {"unloaded": unloaded, "count": len(unloaded), "errors": errors, "ok": len(errors) == 0}


try:
    import folder_paths
except ImportError:
    folder_paths = None


def default_index_root() -> Path:
    if folder_paths and hasattr(folder_paths, "models_dir"):
        root = Path(folder_paths.models_dir) / "RAG" / "VectorDB"
    else:
        root = Path(__file__).resolve().parent / "data" / "faiss_indexes"
    root.mkdir(parents=True, exist_ok=True)
    return root


def index_exists(index_name: str) -> bool:
    index_dir = default_index_root() / index_name
    required_files = ["index.faiss", "chunks.json", "meta.json"]
    return index_dir.exists() and all((index_dir / f).exists() for f in required_files)


# ==============================
# ✅ 【新增】增量索引核心逻辑
#    只对 新增/修改/删除 的文件重新嵌入，未变化文件复用旧向量
# ==============================
def _compute_documents_hashes(documents: List[Dict]) -> Dict[str, str]:
    """为每个文档计算哈希（用于变化检测）。"""
    hashes: Dict[str, str] = {}
    for doc in documents:
        src = doc.get("source", "")
        if not src:
            continue
        p = Path(src)
        hashes[src] = _compute_file_hash(p) if p.exists() else hashlib.md5(
            (doc.get("text") or "").encode()
        ).hexdigest()
    return hashes


def _load_existing_index_data(index_name: str) -> Tuple[List[Dict], np.ndarray, Dict]:
    """
    加载已有索引的 chunks、向量矩阵和 meta。
    返回 (chunks, vectors_np, meta)。
    vectors_np 可能不存在（旧版索引），此时返回空数组并标记需要全量重建。
    """
    index_dir = default_index_root() / index_name
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

    vectors_path = index_dir / "vectors.npy"
    if vectors_path.exists():
        vectors = np.load(str(vectors_path))
    else:
        # 旧版索引没有 vectors.npy，需要全量重建
        vectors = np.zeros((0,), dtype=np.float32)

    return chunks, vectors, meta


def update_faiss_index(
    documents: List[Dict],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    index_name: str,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> Dict:
    """
    ✅ 增量更新索引：
    - 未变化的文件 → 复用已有向量，不重新嵌入
    - 新增/修改的文件 → 重新嵌入
    - 已删除的文件 → 从索引中移除
    - 若旧索引无 vectors.npy（旧版格式）→ 自动回退全量重建
    """
    if not index_exists(index_name):
        return build_faiss_index(documents, embedding_model, chunk_size, chunk_overlap, index_name, batch_size)

    print(f"[EasyRAG] 🔍 检测文件变化，执行增量更新...")

    # 1. 加载已有数据
    old_chunks, old_vectors, old_meta = _load_existing_index_data(index_name)

    # 若旧版索引缺少 vectors.npy，退回全量重建
    if old_vectors.shape[0] == 0 and len(old_chunks) > 0:
        print("[EasyRAG] ⚠️ 旧索引格式不含向量缓存，执行全量重建...")
        return build_faiss_index(documents, embedding_model, chunk_size, chunk_overlap, index_name, batch_size)

    # 若 embedding 模型变了，必须全量重建
    if old_meta.get("embedding_model", "") != embedding_model:
        print(f"[EasyRAG] ⚠️ Embedding 模型已更换（{old_meta.get('embedding_model')} → {embedding_model}），全量重建...")
        return build_faiss_index(documents, embedding_model, chunk_size, chunk_overlap, index_name, batch_size)

    # 2. 计算新旧哈希，找出变化
    old_hashes: Dict[str, str] = old_meta.get("source_hashes", {})
    new_hashes = _compute_documents_hashes(documents)
    new_sources = set(new_hashes.keys())
    old_sources = set(old_hashes.keys())

    added_sources = new_sources - old_sources
    removed_sources = old_sources - new_sources
    changed_sources = {
        s for s in (new_sources & old_sources)
        if new_hashes[s] != old_hashes[s]
    }
    unchanged_sources = new_sources - added_sources - changed_sources

    need_reembed = added_sources | changed_sources
    need_remove = removed_sources | changed_sources

    print(f"[EasyRAG] 📊 变化分析: 新增={len(added_sources)} 修改={len(changed_sources)} "
          f"删除={len(removed_sources)} 未变化={len(unchanged_sources)}")

    # 若全部未变化，直接返回
    if not need_reembed and not need_remove:
        print("[EasyRAG] ✅ 文件无变化，跳过重建。")
        return {
            "index_name": index_name,
            "index_dir": str(default_index_root() / index_name),
            "embedding_model": embedding_model,
            "chunks_count": len(old_chunks),
            "documents_count": len(documents),
            "incremental": True,
            "reembedded": 0,
        }

    # 3. 保留未被移除的旧 chunks 和向量
    kept_chunks: List[Dict] = []
    kept_vectors_list: List[np.ndarray] = []
    for i, chunk in enumerate(old_chunks):
        if chunk.get("source", "") not in need_remove:
            kept_chunks.append(chunk)
            if i < old_vectors.shape[0]:
                kept_vectors_list.append(old_vectors[i])

    # 4. 对新增/修改的文件重新切块并嵌入
    new_chunks: List[Dict] = []
    for doc in documents:
        if doc.get("source", "") not in need_reembed:
            continue
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        split_chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, chunk_text in enumerate(split_chunks):
            new_chunks.append({
                "chunk_id": -1,  # 待重新编号
                "doc_id": -1,
                "source": doc.get("source", ""),
                "title": doc.get("title", ""),
                "text": chunk_text,
                "position": i,
            })

    new_vectors_list: List[np.ndarray] = []
    if new_chunks:
        embedder = EmbeddingBackend(embedding_model)
        new_texts = [c["text"] for c in new_chunks]
        print(f"[EasyRAG] 🧠 重新嵌入 {len(new_texts)} 个 chunk...")
        new_vecs = embedder.encode(new_texts, batch_size=batch_size)
        new_vectors_list = [new_vecs[i] for i in range(new_vecs.shape[0])]
        embedder.release()
        del embedder
        gc.collect()

    # 5. 合并：旧的保留部分 + 新嵌入部分
    all_chunks = kept_chunks + new_chunks
    all_vectors_list = kept_vectors_list + new_vectors_list

    # 重新编号
    for idx, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = idx

    if not all_chunks:
        raise ValueError("更新后索引为空，请检查文档内容。")

    all_vectors = np.vstack(all_vectors_list).astype(np.float32)

    # 6. 保存
    dim = all_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_vectors)

    index_dir = default_index_root() / index_name
    index_dir.mkdir(parents=True, exist_ok=True)

    _faiss_write_index_safe(index, index_dir / "index.faiss")
    (index_dir / "chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # ✅ 保存向量矩阵，供下次增量更新使用
    np.save(str(index_dir / "vectors.npy"), all_vectors)
    (index_dir / "meta.json").write_text(json.dumps({
        "index_name": index_name,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "dim": dim,
        "documents_count": len(documents),
        "chunks_count": len(all_chunks),
        "source_hashes": new_hashes,  # ✅ 保存哈希供下次对比
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[EasyRAG] ✅ 增量更新完成！总 chunk: {len(all_chunks)}，本次重新嵌入: {len(new_chunks)}")

    unload_embedding_model(embedding_model)
    return {
        "index_name": index_name,
        "index_dir": str(index_dir),
        "embedding_model": embedding_model,
        "chunks_count": len(all_chunks),
        "documents_count": len(documents),
        "incremental": True,
        "reembedded": len(new_chunks),
    }


# ==============================
# ✅ 【修改】get_or_create_index
#    新增 force_rebuild 参数；默认走增量更新而非全量重建
# ==============================
def get_or_create_index(
    documents: List[Dict],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    index_name: str,
    force_rebuild: bool = False,
) -> Dict:
    index_dir = default_index_root() / index_name

    if force_rebuild:
        print(f"[EasyRAG] 🔨 强制全量重建索引...")
        result = build_faiss_index(documents, embedding_model, chunk_size, chunk_overlap, index_name)
        unload_embedding_model(embedding_model)
        return result

    if index_exists(index_name):
        # ✅ 索引存在 → 走增量更新（检测变化，只重嵌入变化部分）
        return update_faiss_index(documents, embedding_model, chunk_size, chunk_overlap, index_name)

    print(t("🆘 未找到完整向量库，开始构建..."))
    result = build_faiss_index(documents, embedding_model, chunk_size, chunk_overlap, index_name)
    print(t("✅ 向量库构建完成！"))
    unload_embedding_model(embedding_model)
    return result


def build_faiss_index(
    documents: List[Dict],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    index_name: str,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> Dict:
    if faiss is None:
        raise ImportError("faiss 未安装")
    if not documents:
        raise ValueError("No documents")
    if not index_name.strip():
        raise ValueError("index_name empty")

    embedder = EmbeddingBackend(embedding_model)
    chunks: List[Dict] = []
    for doc_id, doc in enumerate(documents):
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        split_chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, chunk in enumerate(split_chunks):
            chunks.append({
                "chunk_id": len(chunks),
                "doc_id": doc_id,
                "source": doc.get("source", ""),
                "title": doc.get("title", ""),
                "text": chunk,
                "position": i,
            })

    if not chunks:
        raise ValueError("No chunks")

    chunk_texts = [x["text"] for x in chunks]
    print(f"[EasyRAG] 🧠 全量嵌入 {len(chunk_texts)} 个 chunk（batch_size={batch_size}）...")
    vectors = embedder.encode(chunk_texts, batch_size=batch_size)
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    root = default_index_root()
    index_dir = root / index_name
    index_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 计算并保存文件哈希（供增量更新使用）
    source_hashes = _compute_documents_hashes(documents)

    _faiss_write_index_safe(index, index_dir / "index.faiss")
    (index_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # ✅ 保存向量矩阵（供增量更新复用）
    np.save(str(index_dir / "vectors.npy"), vectors)
    (index_dir / "meta.json").write_text(json.dumps({
        "index_name": index_name,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "dim": dim,
        "documents_count": len(documents),
        "chunks_count": len(chunks),
        "source_hashes": source_hashes,  # ✅ 新增
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    embedder.release()
    del embedder
    gc.collect()

    return {
        "index_name": index_name,
        "index_dir": str(index_dir),
        "embedding_model": embedding_model,
        "chunks_count": len(chunks),
        "documents_count": len(documents),
    }


def load_index(index_name_or_path: str) -> Tuple[Any, List[Dict], Dict]:
    if faiss is None:
        raise ImportError("faiss not installed")
    path = Path(index_name_or_path)
    if path.is_dir():
        index_dir = path
    else:
        index_dir = default_index_root() / index_name_or_path
    if not index_dir.exists():
        raise FileNotFoundError(f"Index not found: {index_dir}")

    index = _faiss_read_index_safe(index_dir / "index.faiss")
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    return index, chunks, meta


def search_index(
    index_name_or_path: str,
    query: str,
    top_k: int = 5,
    device: Optional[str] = None,
) -> Dict:
    if not query.strip():
        raise ValueError("query empty")

    top_k = max(1, int(top_k))
    index, chunks, meta = load_index(index_name_or_path)
    embedder = EmbeddingBackend(meta["embedding_model"], device=device)
    qvec = embedder.encode([query])
    scores, indices = index.search(qvec, top_k)

    items: List[Dict] = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        items.append({
            "score": float(score),
            "text": chunk["text"],
            "source": chunk.get("source", ""),
            "title": chunk.get("title", ""),
            "position": chunk.get("position", 0),
        })

    context_lines: List[str] = []
    for i, item in enumerate(items, start=1):
        context_lines.append(f"[{i}] score={item['score']:.4f}\n{item['text']}")

    embedder.release()
    del embedder
    gc.collect()

    return {
        "query": query,
        "top_k": top_k,
        "items": items,
        "rag_hit": len(items) > 0,
        "best_score": items[0]["score"] if items else 0.0,
        "context": "\n\n".join(context_lines).strip(),
    }


def resolve_lmstudio_model(base_url: str, timeout: int = 20) -> str:
    models = list_lmstudio_models(base_url=base_url, timeout=timeout)
    if not models:
        raise RuntimeError("LM Studio 模型列表为空")
    return models[0]


def list_lmstudio_models(base_url: str, timeout: int = 10) -> List[str]:
    base = base_url.rstrip("/")
    out = []
    try:
        resp = requests.get(base + "/api/v1/models", timeout=timeout)
        if resp.ok:
            data = resp.json()
            for m in data.get("models", []):
                key = m.get("key") or m.get("id")
                if key:
                    out.append(str(key))
    except Exception:
        pass
    if not out:
        try:
            resp = requests.get(base + "/v1/models", timeout=timeout)
            if resp.ok:
                data = resp.json()
                for m in data.get("data", []):
                    mid = m.get("id")
                    if mid:
                        out.append(str(mid))
        except Exception:
            pass
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def unload_lmstudio_model(base_url: str, instance_id: str, timeout: int = 20) -> Dict:
    ep = base_url.rstrip("/") + "/api/v1/models/unload"
    r = requests.post(ep, json={"instance_id": instance_id}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _normalize_text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", "") or item.get("content", ""))
        return "\n".join(p.strip() for p in parts if p.strip())
    if isinstance(value, dict):
        return _normalize_text_content(value.get("text") or value.get("content"))
    return str(value).strip()


def _pick_answer(content_text: str, reasoning_text: str) -> str:
    return content_text or reasoning_text


def _extract_answer_from_chat_payload(data: Dict) -> Dict:
    msg = data.get("choices", [{}])[0].get("message", {}) if isinstance(data, dict) else {}
    c = _normalize_text_content(msg.get("content"))
    r = _normalize_text_content(msg.get("reasoning_content") or msg.get("reasoning"))
    return {"answer": _pick_answer(c, r).strip(), "content_text": c, "reasoning_text": r}


def _extract_answer_from_responses_payload(data: Dict) -> Dict:
    c_parts = []
    r_parts = []
    ot = _normalize_text_content(data.get("output_text"))
    if ot:
        c_parts.append(ot)
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        t = str(item.get("type", "")).lower()
        if t == "message":
            for cont in item.get("content", []):
                ct = str(cont.get("type", "")).lower()
                txt = _normalize_text_content(cont.get("text"))
                if not txt:
                    continue
                if ct in ("output_text", "text"):
                    c_parts.append(txt)
                elif "reasoning" in ct:
                    r_parts.append(txt)
        elif "reasoning" in t:
            txt = _normalize_text_content(item.get("reasoning_content") or item.get("text") or item.get("content"))
            if txt:
                r_parts.append(txt)
    rt = _normalize_text_content(data.get("reasoning_content"))
    if rt:
        r_parts.append(rt)
    c = "\n".join(c_parts).strip()
    r = "\n".join(r_parts).strip()
    return {"answer": _pick_answer(c, r).strip(), "content_text": c, "reasoning_text": r}


def _stream_chat_completions(ep, payload, timeout, emit, headers=None):
    c_parts = []
    r_parts = []
    s_parts = []
    with requests.post(ep, json=payload, headers=headers, stream=True, timeout=timeout) as resp:
        if not resp.ok:
            error_detail = ""
            try:
                error_detail = resp.text
            except:
                pass
            raise RuntimeError(f"HTTP {resp.status_code}: {error_detail or resp.reason}")
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=False):
            if not line:
                continue
            s = line.decode("utf-8", "ignore").strip()
            if not s.startswith("data:"):
                continue
            d = s[5:].strip()
            if d == "[DONE]":
                break
            try:
                e = json.loads(d)
            except Exception:
                continue
            delta = e.get("choices", [{}])[0].get("delta", {})
            c = _normalize_text_content(delta.get("content"))
            r = _normalize_text_content(delta.get("reasoning_content") or delta.get("reasoning"))
            if c:
                c_parts.append(c)
                s_parts.append(c)
                if emit:
                    print(c, end="", flush=True)
            if r:
                r_parts.append(r)
                s_parts.append(r)
                if emit:
                    print(r, end="", flush=True)
    if emit and s_parts:
        print()
    return {
        "answer": _pick_answer("".join(c_parts), "".join(r_parts)).strip(),
        "content_text": "".join(c_parts).strip(),
        "reasoning_text": "".join(r_parts).strip(),
        "stream_text": "".join(s_parts).strip(),
        "raw": {"stream": True, "api_mode": "chat"}
    }


def _stream_responses(ep, payload, timeout, emit):
    ev = ""
    c = []
    r = []
    s = []
    final = {}
    with requests.post(ep, json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=False):
            if not line:
                continue
            u = line.decode("utf-8", "ignore").strip()
            if u.startswith("event:"):
                ev = u[6:].strip()
                continue
            if not u.startswith("data:"):
                continue
            d = u[5:].strip()
            if d == "[DONE]":
                break
            try:
                dat = json.loads(d)
            except Exception:
                continue
            if ev.endswith(".completed") and isinstance(dat, dict):
                final = dat.get("response", dat)
            dt = _normalize_text_content(dat.get("delta"))
            if not dt:
                continue
            if "reasoning" in ev:
                r.append(dt)
            else:
                c.append(dt)
            s.append(dt)
            if emit:
                print(dt, end="", flush=True)
    if emit and s:
        print()
    cc = "".join(c).strip()
    rr = "".join(r).strip()
    if final:
        ext = _extract_answer_from_responses_payload(final)
        cc = ext["content_text"] or cc
        rr = ext["reasoning_text"] or rr
    ans = _pick_answer(cc, rr).strip()
    return {
        "answer": ans,
        "content_text": cc,
        "reasoning_text": rr,
        "stream_text": "".join(s).strip(),
        "raw": final or {"stream": True, "api_mode": "responses"}
    }


def lmstudio_chat(
    base_url: str,
    model: str,
    question: str,
    context: str = "",
    image_data_url: str = "",
    system_prompt: str = "You are a helpful assistant.",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    api_mode: str = "responses",
    stream: bool = False,
    emit_stream_log: bool = False,
    timeout: int = 120,
) -> Dict:
    if not model.strip():
        model = resolve_lmstudio_model(base_url)
    q = question.strip()
    if context.strip():
        q = t("请根据上下文回答：\n{context}\n\n问题：{question}", context=context.strip(), question=question.strip())
    mode = api_mode.strip().lower()
    base = base_url.rstrip("/")
    if mode == "responses":
        ep = base + "/v1/responses"
        inp = [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": q}]}
        ]
        if image_data_url.strip():
            inp[1]["content"].append({"type": "input_image", "image_url": image_data_url.strip()})
        payload = {"model": model, "input": inp, "stream": stream}
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_output_tokens"] = int(max_tokens)
        if seed is not None:
            payload["seed"] = int(seed)
    else:
        ep = base + "/v1/chat/completions"
        msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
        if image_data_url.strip():
            msg[1]["content"] = [
                {"type": "text", "text": q},
                {"type": "image_url", "image_url": {"url": image_data_url.strip()}}
            ]
        payload = {"model": model, "messages": msg, "stream": stream}
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if seed is not None:
            payload["seed"] = int(seed)
    try:
        if stream:
            if mode == "responses":
                res = _stream_responses(ep, payload, timeout, emit_stream_log)
            else:
                res = _stream_chat_completions(ep, payload, timeout, emit_stream_log)
            return {"answer": res["answer"], "raw": res["raw"], "model": model, "stream_text": res["stream_text"]}
        resp = requests.post(ep, json=payload, timeout=timeout)
    except Exception as e:
        if mode == "responses":
            return lmstudio_chat(base_url, model, question, context, image_data_url, system_prompt,
                                  temperature, max_tokens, "chat_completions", stream, emit_stream_log, timeout)
        raise RuntimeError(f"API 连接失败：{e}")
    resp.raise_for_status()
    data = resp.json()
    if mode == "responses":
        ext = _extract_answer_from_responses_payload(data)
    else:
        ext = _extract_answer_from_chat_payload(data)

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"answer": ext["answer"].strip(), "raw": data, "model": model, "stream_text": ext["answer"].strip()}


def external_api_chat(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    context: str = "",
    image_data_url: str = "",
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    stream: bool = False,
    emit_stream_log: bool = False,
    timeout: int = 120,
) -> Dict:
    q = question.strip()
    if context.strip():
        q = t("请根据上下文回答：\n{context}\n\n问题：{question}", context=context.strip(), question=question.strip())

    base = base_url.rstrip("/")
    if not base.endswith("/v1") and not base.endswith("/v1/"):
        ep = base + "/v1/chat/completions"
    elif base.endswith("/v1"):
        ep = base + "/chat/completions"
    else:
        ep = base + "chat/completions"

    if "/chat/completions" in base_url:
        ep = base_url

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
    if image_data_url.strip():
        msg[1]["content"] = [
            {"type": "text", "text": q},
            {"type": "image_url", "image_url": {"url": image_data_url.strip()}}
        ]

    payload = {
        "model": model,
        "messages": msg,
        "stream": stream,
        "temperature": temperature,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens

    if seed is not None and seed > 0:
        payload["seed"] = seed

    try:
        if stream:
            res = _stream_chat_completions(ep, payload, timeout, emit_stream_log, headers=headers)
            return {"answer": res["answer"], "raw": res["raw"], "model": model, "stream_text": res["stream_text"]}

        resp = requests.post(ep, json=payload, headers=headers, timeout=timeout)
        if not resp.ok:
            error_detail = ""
            try:
                error_detail = resp.text
            except:
                pass
            raise RuntimeError(f"HTTP {resp.status_code}: {error_detail or resp.reason}")

        data = resp.json()
        ext = _extract_answer_from_chat_payload(data)

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"answer": ext["answer"].strip(), "raw": data, "model": model, "stream_text": ext["answer"].strip()}
    except Exception as e:
        raise RuntimeError(f"API 调用失败：{e}")


def extract_answer_between_newlines(content: str) -> str:
    text = (content or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            inner = "\n".join(lines[1:-1]).strip()
            if inner:
                return inner
    return text
