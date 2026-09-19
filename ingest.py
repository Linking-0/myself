"""构建 lin43 的个人知识库：加载 → 分块 → embedding → 写入 Chroma。

使用方式：
    python ingest.py                 # 完整重建
    python ingest.py --reset         # 清空后重建
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_MODEL,
    SOURCE_FILES,
)


# ---------- 分块工具 ----------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按字符数切分，overlap 保证块间不切断句子。"""
    if not text.strip():
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def stable_id(text: str) -> str:
    """为每个 chunk 生成稳定 hash id，便于增量更新时去重。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ---------- JSONL 对话摘要 ----------

def load_jsonl_summaries(path: Path) -> list[str]:
    """解析 session_memory_*.jsonl，每条摘要本身就是一个完美 chunk。

    每条记录结构：
      intent        用户意图
      actions[]     AI 做了什么
      learned[]     这轮学到的（最有价值）
      outcome       结果如何
      message_summary_time  时间戳
    """
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        parts = []
        ts = rec.get("message_summary_time", "?")
        parts.append(f"[{ts}]")

        intent = rec.get("intent", "").strip()
        if intent:
            parts.append(f"意图: {intent}")

        actions = rec.get("actions", [])
        if actions:
            parts.append("AI做了什么:")
            for a in actions:
                parts.append(f"  - {a}")

        learned = rec.get("learned", [])
        if learned:
            parts.append("学到的:")
            for l in learned:
                parts.append(f"  - {l}")

        outcome = rec.get("outcome", "").strip()
        if outcome:
            parts.append(f"结果: {outcome}")

        chunk = "\n".join(parts)
        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ---------- 主流程 ----------

def build_kb(reset: bool = False) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 加载 embedding 模型：{EMBED_MODEL}  ...", end=" ", flush=True)
    model = SentenceTransformer(EMBED_MODEL)
    print("OK")

    print(f"[2/3] 连接 Chroma @ {CHROMA_DIR}  ...", end=" ", flush=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print("OK")

    # 收集所有文档
    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_docs: list[str] = []
    all_metas: list[dict] = []

    loaded = 0
    skipped = 0
    for src_path, category in SOURCE_FILES:
        if not src_path.exists():
            print(f"  SKIP (不存在): {src_path}")
            skipped += 1
            continue

        # 按文件类型加载
        if src_path.suffix == ".jsonl":
            # 对话摘要：每条记录就是一个 chunk，不需要再切
            chunks = load_jsonl_summaries(src_path)
        else:
            # 普通文本：读出来再分块
            text = src_path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_text(text)

        if not chunks:
            continue

        # embedding
        embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

        rel = src_path.relative_to(Path.cwd()) if src_path.is_absolute() and str(src_path).startswith(str(Path.cwd())) else src_path
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cid = f"{category}-{stable_id(chunk)}-{i}"
            all_ids.append(cid)
            all_embeddings.append(emb.tolist())
            all_docs.append(chunk)
            all_metas.append({
                "category": category,
                "source": str(rel),
                "chunk_index": i,
            })

        loaded += 1
        print(f"  + [{category}] {rel}  → {len(chunks)} chunks")

    print(f"[3/3] 写入 Chroma  ...", end=" ", flush=True)
    if all_ids:
        # 分批写入，避免单次 payload 过大
        BATCH = 200
        for i in range(0, len(all_ids), BATCH):
            j = min(i + BATCH, len(all_ids))
            collection.add(
                ids=all_ids[i:j],
                embeddings=all_embeddings[i:j],
                documents=all_docs[i:j],
                metadatas=all_metas[i:j],
            )
    print(f"OK  ({len(all_ids)} chunks from {loaded} files, skipped {skipped})")
    print(f"\n知识库位置：{CHROMA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="清空旧库后重建")
    args = parser.parse_args()
    build_kb(reset=args.reset)
