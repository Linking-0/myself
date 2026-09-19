"""lin43 个人 Agent：检索 + DeepSeek 聊天。

使用方式：
    # 先确保 .env 里有 DEEPSEEK_API_KEY
    python agent.py                  # 启动交互 REPL
    python agent.py "STM32 PA3 接哪根线？"   # 直接问一句退出
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# UTF-8 兜底：Streamlit Cloud / 某些 Linux 容器默认 locale 不是 UTF-8，
# print 中文会抛 UnicodeEncodeError
if sys.stdout.encoding not in ("utf-8", "UTF-8"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr.encoding not in ("utf-8", "UTF-8"):
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI
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

        # DeepSeek（OpenAI 兼容 SDK）
        self.llm = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

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

        if stream:
            # 流式模式（REPL 用）
            stream_resp = self.llm.chat.completions.create(
                model=self.model, messages=messages, stream=True, temperature=0.3,
            )
            chunks = []
            for delta in stream_resp:
                if delta.choices and delta.choices[0].delta.content:
                    text = delta.choices[0].delta.content
                    chunks.append(text)
                    print(text, end="", flush=True)
            print()
            answer = "".join(chunks)
        else:
            # 非流式（单次问答用）
            resp = self.llm.chat.completions.create(
                model=self.model, messages=messages, stream=False, temperature=0.3,
            )
            answer = resp.choices[0].message.content or ""

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
