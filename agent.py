"""lin43 个人 Agent：检索 + DeepSeek 聊天。

使用方式：
    # 先确保 .env 里有 DEEPSEEK_API_KEY
    python agent.py                  # 启动交互 REPL
    python agent.py "STM32 PA3 接哪根线？"   # 直接问一句退出
"""
from __future__ import annotations

import io
import locale
import os
import sys
from pathlib import Path

# ===== 编码炸弹：Streamlit Cloud / 某些 Linux 容器 =====
# 默认 locale 是 POSIX / C，openai SDK 序列化中文时抛 UnicodeEncodeError
# 必须在一切第三方 import 之前修掉
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("LANG", "C.UTF-8")
os.environ.setdefault("LC_ALL", "C.UTF-8")
try:
    locale.setlocale(locale.LC_ALL, "C.UTF-8")
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    except locale.Error:
        pass

# UTF-8 兜底 stdout/stderr
for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream is not None and hasattr(stream, "buffer"):
        enc = getattr(stream, "encoding", "") or ""
        if enc.lower() != "utf-8":
            try:
                setattr(sys, stream_name, io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace"))
            except Exception:
                pass

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEEPSEEK_BASE_URL,
    EMBED_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
)


def _check_env() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-xxxxxxxx"):
        print("❌ 没找到有效的 DEEPSEEK_API_KEY。")
        print(f"   请复制 .env.example 为 .env 并填入你的 key。")
        sys.exit(1)
    return api_key


class Lin43Agent:
    def __init__(self) -> None:
        load_dotenv()
        api_key = _check_env()

        # Embedding（和 ingest.py 用同一个模型，保证向量空间一致）
        self.embedder = SentenceTransformer(EMBED_MODEL)

        # Chroma
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        # 知识库为空时自动构建（HF Spaces 首次启动、本地删了 chroma_db 都会触发）
        try:
            self.collection = client.get_collection(COLLECTION_NAME)
            kb_count = self.collection.count()
        except Exception:
            kb_count = 0

        if kb_count == 0:
            print("🗄️  知识库为空，自动运行 ingest.py 构建...", flush=True)
            from ingest import build_kb
            build_kb(reset=True)
            self.collection = client.get_collection(COLLECTION_NAME)
            print(f"✅ 知识库就绪：{self.collection.count()} 块", flush=True)

        # DeepSeek：直接用 requests（绕过 openai SDK 在 Python 3.14 上的编码 bug）
        self.api_key = api_key
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.api_url = f"{DEEPSEEK_BASE_URL}/chat/completions"

        # 对话历史（内存中，单轮不需要多轮上下文也能工作）
        self.history: list[dict] = []

    # ---- 检索 ----
    def _retrieve(self, query: str) -> list[dict]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append({"id": cid, "doc": doc, "meta": meta, "dist": dist})
        return hits

    # ---- 回答 ----
    def ask(self, question: str, show_context: bool = False, stream: bool = True) -> str:
        hits = self._retrieve(question)

        # 拼检索上下文
        if hits:
            context_parts = []
            for i, h in enumerate(hits):
                src = h["meta"].get("source", "?")
                cat = h["meta"].get("category", "?")
                context_parts.append(f"--- [{i+1}] {cat} | {src} ---\n{h['doc']}")
            context = "\n\n".join(context_parts)
        else:
            context = "（知识库中没找到相关片段，仅基于我对 lin43 的认知回答）"

        # 系统消息 + 本轮上下文注入
        user_msg = (
            f"## 用户问题\n{question}\n\n"
            f"## 知识库检索片段（仅供参考，可能不完整）\n{context}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history,
            {"role": "user", "content": user_msg},
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "stream": stream,
        }

        if stream:
            # 流式模式（REPL 用）
            chunks = []
            with requests.post(self.api_url, headers=headers, json=payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    line = line[6:]
                    if line == "[DONE]":
                        break
                    try:
                        import json as _json
                        d = _json.loads(line)
                        text = d["choices"][0]["delta"].get("content", "")
                        if text:
                            chunks.append(text)
                            print(text, end="", flush=True)
                    except Exception:
                        continue
            print()
            answer = "".join(chunks)
        else:
            # 非流式（Streamlit 用）
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"] or ""

        # 追加到历史
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        # 可选：显示检索来源
        if show_context and hits:
            print("\n--- 检索来源 ---")
            for h in hits:
                src = h["meta"].get("source", "?")
                cat = h["meta"].get("category", "?")
                dist = h["dist"]
                print(f"  [{cat}] {src}  (距离={dist:.3f})")

        return answer

    def reset(self) -> None:
        self.history.clear()
        print("🔄 对话历史已清空")


# ---- REPL ----
def repl(stream: bool = False) -> None:
    agent = Lin43Agent()
    print(f"\n🤖 lin43 Agent 已就绪（模型={agent.model}，kb={agent.collection.count()} 块）")
    print(f"   模式：{'流式' if stream else '非流式'}（加 --stream 可切流式）")
    print("   /sources 显示检索来源  /reset 清空历史  /exit 退出\n")

    while True:
        try:
            q = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见")
            break

        if not q:
            continue
        if q in ("/exit", "/quit", "exit", "quit"):
            print("👋 再见")
            break
        if q == "/reset":
            agent.reset()
            continue

        show = q.endswith(" /sources")
        question = q.replace(" /sources", "")
        if not stream:
            print("Agent > ", end="", flush=True)
        answer = agent.ask(question, show_context=show, stream=stream)
        if not stream:
            print(answer)
            print()


if __name__ == "__main__":
    # 解析 --stream / --no-stream 开关
    args = sys.argv[1:]
    stream = "--stream" in args
    args = [a for a in args if a != "--stream"]

    if args:
        # 单次问答模式
        agent = Lin43Agent()
        q = " ".join(args)
        answer = agent.ask(q, stream=stream)
        if not stream:
            print(answer)
    else:
        repl(stream=stream)
