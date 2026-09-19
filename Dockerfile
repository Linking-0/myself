FROM python:3.11-slim

# HF Spaces 要求：应用放在 /app 下，/data 是持久化目录
WORKDIR /app

# 先拷 requirements（利用 Docker 缓存层）
COPY requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt

# 拷全部代码
COPY *.py .
COPY personal.md .

# HF Spaces 启动 Streamlit
EXPOSE 7860
CMD ["streamlit", "run", "agent_web.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
