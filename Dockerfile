# MetaForge — 可复现运行镜像 (作者: 晨星)
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

# 再拷源码
COPY metaforge ./metaforge
COPY tests ./tests
COPY pyproject.toml .
COPY metaforge/examples ./metaforge/examples

# 校验：跑全部测试（含最关键的梯度校验），失败则构建失败
RUN python -m pytest -q -W ignore::UserWarning

# 默认：端到端演示，生成真实基线
CMD ["python", "-m", "metaforge.examples.run_demo"]
