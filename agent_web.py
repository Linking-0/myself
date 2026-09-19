"""lin43 个人 Agent 的 Streamlit 网页前端。

本地运行：
    pip install streamlit
    streamlit run agent_web.py

部署到 HuggingFace Spaces 时，DEEPSEEK_API_KEY 通过 HF Secret 注入。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 确保能 import config / agent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from dotenv import load_dotenv

# 加载本地 .env（本地开发用；HF Spaces 通过 Secret 注入）
load_dotenv()

from agent import Lin43Agent


# ---- 页面配置 ----
st.set_page_config(
    page_title="lin43 的 AI 分身",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 千然的 AI 分身")
st.caption("由他的资料、项目文档、真实对话蒸馏而成。可以问他关于自己的事，也可以问技术问题。")

# ---- API Key 检查 ----
api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
if not api_key:
    st.error(
        "❌ 没找到 DEEPSEEK_API_KEY。\n\n"
        "本地运行：在 `personal_agent/.env` 里填。\n"
        "HuggingFace Spaces：在 Settings → Secrets and variables 里设。"
    )
    st.stop()

# ---- 缓存 Agent（HF Spaces 上加载 torch 要几秒，缓存住） ----
@st.cache_resource(show_spinner=False)
def get_agent() -> Lin43Agent:
    print("[agent_web] 初始化 Lin43Agent (首次加载，后续会缓存)...", flush=True)
    return Lin43Agent()


with st.spinner("正在加载千然的知识和记忆（首次约 10 秒）..."):
    try:
        agent = get_agent()
    except Exception as e:
        st.error(f"加载失败：{e}")
        st.stop()

# ---- 侧边栏 ----
with st.sidebar:
    st.subheader("💡 使用提示")
    st.markdown("""
        - 跟他聊聊他是谁、喜欢什么
        - 问技术问题（STM32、继电器接线、OpenCV）
        - 他记得自己的决策历史（87 条摘要）
    """)

    st.subheader("⚙️ 选项")
    show_sources = st.checkbox("显示知识库来源", value=False, help="回答下方会列出引用了哪些资料")
    if st.button("🗑️ 清空对话历史"):
        agent.reset()
        st.success("已清空")

    st.divider()
    st.caption(f"知识库：{agent.collection.count()} 块")
    st.caption(f"模型：{agent.model}")

# ---- 初始化 chat 历史（streamlit 会话）----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- 渲染历史消息 ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 参考来源"):
                for s in msg["sources"]:
                    st.markdown(f"- **[{s['meta']['category']}]** {s['meta']['source']}  (距离={s['dist']:.3f})")

# ---- 接收用户输入 ----
if question := st.chat_input("跟千然聊聊..."):
    # 展示用户消息
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    # Agent 回答
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("千然正在想..."):
            # 用非流式（HF 上流式显示有兼容问题）
            answer = agent.ask(question, stream=False)
            st.markdown(answer)

            sources = []
            if show_sources:
                hits = agent._retrieve(question)  # 内部方法，拿到引用来源
                sources = hits
                with st.expander("📚 参考来源"):
                    for h in hits:
                        src = h["meta"].get("source", "?")
                        cat = h["meta"].get("category", "?")
                        st.markdown(f"- **[{cat}]** {src}  (距离={h['dist']:.3f})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
