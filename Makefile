# MetaForge Makefile — 作者: 晨星
# 用法: make <target>

PY ?= python
VENV ?= .venv

.PHONY: help install dev test lint format check demo benchmark clean

help:
	@echo "MetaForge targets:"
	@echo "  install    创建 venv 并安装锁定依赖"
	@echo "  dev        安装运行时+开发依赖(宽松)"
	@echo "  test       运行完整 pytest 套件"
	@echo "  lint       ruff check"
	@echo "  format     ruff format"
	@echo "  check      lint + format --check (CI 门槛)"
	@echo "  demo       端到端示例，生成真实基线"
	@echo "  benchmark  命令行全量基准"
	@echo "  clean      清理缓存"

install:
	$(PY) -m venv $(VENV)
	$(VENV)/Scripts/pip install -r requirements.lock.txt  # Windows
	# 若使用 Linux/macOS, 改用: $(VENV)/bin/pip install -r requirements.lock.txt

dev:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q -W ignore::UserWarning

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

check: lint
	$(PY) -m ruff format --check .

demo:
	$(PY) -m metaforge.examples.run_demo

benchmark:
	$(PY) -m metaforge.cli --domain all --out benchmark.json

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache $(VENV)
	find . -name "*.pyc" -delete
