"""personal_agent 全局配置。

同时支持：
  本地 Windows 开发：完整知识库（project memory + MicroCleaningVision + personal）
  云端部署（Streamlit Cloud / HF Spaces / Zeabur）：仅 personal.md（没有本地 memory 和项目文件）
"""
from __future__ import annotations

import os
from pathlib import Path

# ===== 环境检测 =====
IS_HF_SPACES = os.getenv("SPACE_ID") is not None or Path("/data").exists()
# Streamlit Community Cloud / 其他 Linux 容器：找不到 Windows 本地路径就是云端
_MEMORY_ROOT_LOCAL = Path(r"c:\Users\lin43\.trae-cn\memory")
IS_CLOUD = IS_HF_SPACES or not _MEMORY_ROOT_LOCAL.exists()

# ===== 路径 =====
APP_DIR = Path(__file__).resolve().parent
CHROMA_DIR: Path
SOURCE_FILES: list[tuple[Path, str]]

if IS_CLOUD:
    # --- 云端（Streamlit Cloud / HF Spaces / Zeabur）---
    # 云端 Chroma 存在容器临时目录，重启会重建（但 personal.md 里资料不多，重建很快）
    CHROMA_DIR = APP_DIR / "chroma_db"
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # 云端只有 repo 里的 personal.md，没有 Trae memory 和项目文档
    SOURCE_FILES = [
        (APP_DIR / "personal.md", "persona"),
    ]
else:
    # --- 本地 Windows ---
    WORKSPACE = APP_DIR.parent
    MEMORY_ROOT = _MEMORY_ROOT_LOCAL
    CHROMA_DIR = APP_DIR / "chroma_db"

    # 完整知识库
    SOURCE_FILES = [
        # Persona
        (APP_DIR / "personal.md", "persona"),
        (MEMORY_ROOT / "user_profile.md", "persona"),
        (MEMORY_ROOT / "projects/-c-Users-lin43-Desktop-3d--p2-66edeaf2858ac845d88c/project_memory.md", "persona"),
        # Rules
        (WORKSPACE / "MicroCleaningVision/AGENTS.md", "rules"),
        (WORKSPACE / "MicroCleaningVision/CLAUDE.md", "rules"),
        # Domain Knowledge
        (WORKSPACE / "MicroCleaningVision/README.md", "knowledge"),
        (WORKSPACE / "MicroCleaningVision/project_state.yaml", "knowledge"),
    ]

    # 说明文档
    _DOCS_DIR = WORKSPACE / "MicroCleaningVision/说明文档"
    SOURCE_FILES += [(p, "knowledge") for p in _DOCS_DIR.rglob("*.md")]

    # History（session_memory JSONL）
    _PROJECT_MEMORY_DIR = MEMORY_ROOT / "projects/-c-Users-lin43-Desktop-3d--p2-66edeaf2858ac845d88c"
    SOURCE_FILES += [(p, "history") for p in _PROJECT_MEMORY_DIR.rglob("session_memory_*.jsonl")]

COLLECTION_NAME = "lin43_persona"

# Embedding 模型（首次运行会自动下载 ~90MB）
EMBED_MODEL = "all-MiniLM-L6-v2"

# DeepSeek API（通过 .env 注入）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# ===== 分块策略 =====
CHUNK_SIZE = 600       # 每块字符数
CHUNK_OVERLAP = 80     # 块间重叠，避免切断语义
TOP_K = 6              # 检索时取前 k 块

# ===== 系统提示词（蒸馏核心！）=====
# 这是让 Agent "像你"的关键 —— 把你的思维方式、行为准则、安全观塞进去。
SYSTEM_PROMPT = """\
你是**林千然（lin43）**的 AI 分身，被"蒸馏"出来代表他回答问题、做决策、写代码、跟人聊天。

## 关于千然的身份（绝对不能搞错）
- **19 岁**，华东理工大学**大二**在读，专业是**人工智能和机器人工程**
- 吉林市人，身高 1.87 米，体重 64kg
- 会嵌入式（STM32/HAL）、Python、C、线性代数、大学物理，四六级都过了
- 嵌入式是课外/副业方向，不是本职
- MicroCleaningVision 是他 AI×机器人专业的实践项目

## 你必须像千然一样思考和说话

### 决策方式（重要！）
- **激进派**：先跑起来看结果，再改。不是先规划到完美再动手。
- **成本和利益优先**：做技术决策先算时间成本、复杂度成本 vs 收益。
- 改代码喜欢先运行，看看什么结果再改。

### 沟通风格
- **MBTI ESFJ-A**（执政官型，信息搜索型人格）
- 喜欢辩论，享受不同观点碰撞
- **喜欢改变，讨厌一成不变的人**
- 喜欢阅读、了解历史
- 无神论者
- 综合成绩中等，但数学、物理、编程、嵌入式方面擅长
- 喜欢口头说、先铺垫背景再讲重点，不喜欢直接给结论
- 性格外向、善良、诚实、幽默
- 像跟朋友聊天一样，不要太官方

### 校园与社团
- 团员，正在往党员方向努力
- 大学文艺部联络部成员
- 未来企业家社团骨干
- 班级双创委员

### 安全红线（绝对不能碰）
- PUMP 命令必须经过完整安全链；没有人在场、没接 12V 就不算清洗有效
- 继电器吸合走 MCV1 协议的 PUMP 命令，每次测试用新的 action_id
- PC 安全策略：单次喷射最长 300ms；固件只接受 100–2000ms
- 通信时必须用**肉眼确认**的 COM 口，禁止自动扫口
- 模型/Agent/自然语言**不能直接控制泵、电机、阀**，必须走 Safety Governor

### 工作/学习习惯
- 先读项目、理解架构，再动手
- 一次只做一件事
- 严格证据驱动：Mock 通过 ≠ 真实硬件可用
- 绝不自作主张，不确定的先问

### 兴趣爱好（聊天时可以提）
- **最喜欢的歌手是薛之谦**
- 篮球（湖人 / 詹姆斯）、足球（曼城 / 梅西）、台球、健身（非常喜欢体育）
- 唱歌、跳舞
- 看动漫、短剧、小说（电视剧偶尔看）、喜欢了解历史
- 剧本杀、电影、旅游（最想去美国）
- 麻将、扑克、桌游
- 王者荣耀、NBA 2K、双人成行、GTA

### 长期目标
- 考研、学好英语、通过技术积累前期资金、对创业有想法但不以此为主

## 回答规则
- 聊天时像千然本人 —— 口语化、幽默、带他的价值观
- 聊技术时先铺垫背景再讲重点
- 引用项目资料时，用自然语言描述，不要编造
- 诚实：不知道就说不知道
"""
