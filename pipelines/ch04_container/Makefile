# 4장 Makefile. Docker가 있는 컴퓨터에서만 build/run이 돈다. 테스트와 실험은 Docker 없이 돈다.
IMAGE ?= math-ai-matrix-labs/ch04:local

.PHONY: build run digest test experiments lock

build:
	docker build -t $(IMAGE) .

run:
	docker run --rm $(IMAGE)

digest:
	docker image inspect --format "{{.Id}}" $(IMAGE)

test:
	cd ../.. && uv run pytest -q pipelines/ch04_container

experiments:
	cd ../.. && uv run python pipelines/ch04_container/experiments.py

lock:
	printf "numpy==2.3.3\npytest==8.4.2\n" | uv pip compile - --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes -q --no-header > requirements.lock
