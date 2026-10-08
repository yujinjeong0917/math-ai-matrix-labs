# syntax=docker/dockerfile:1
# 4장 이미지. 이 장에서는 Docker 데몬을 띄우지 않았다. 규칙은 test_pipelines_ch04_container.py가 검사하고,
# 빌드 원리는 pipelines_ch04_image.py가 흉내 낸다. 실제로 빌드하려면 `make build`.
# 베이스: python:3.12-slim 태그가 2026-10-08에 가리키던 다이제스트(Docker Hub 레지스트리에서 확인).
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=2
WORKDIR /app
# 1) 의존성 목록만 먼저 복사하고 설치한다. 소스를 고쳐도 이 두 층은 캐시에서 나온다.
COPY requirements.lock /app/
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
# 2) 소스는 그다음에.
COPY pipelines_ch04_data.py pipelines_ch04_model_np.py pipelines_ch04_job.py /app/
USER 10001
ENTRYPOINT ["python", "pipelines_ch04_job.py"]
