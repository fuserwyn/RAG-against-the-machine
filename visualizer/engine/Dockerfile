# rag-engine: the project's Python stack + the trace server.
# The project itself (src/ + data/) is mounted read-only at /app/project.
FROM ghcr.io/astral-sh/uv:0.11.30 AS uv

FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /build
# lock file of the project, passed as a named build context
COPY --from=project pyproject.toml uv.lock ./
# same versions as uv.lock, but CPU-only torch (no 3 GB of CUDA wheels)
RUN uv venv /opt/venv --python /usr/local/bin/python3 \
 && uv export --frozen --no-dev --no-hashes --no-emit-project \
      --no-emit-package torch -o req.txt \
 && grep -v -E '^(nvidia-|triton|cuda-)' req.txt > req-cpu.txt \
 && uv pip install --python /opt/venv/bin/python \
      --index-url https://download.pytorch.org/whl/cpu torch \
 && uv pip install --python /opt/venv/bin/python -r req-cpu.txt \
 && rm -rf /root/.cache/uv

COPY trace_server.py inside.py /srv/
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/project
WORKDIR /app/project
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=20 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"
CMD ["python", "/srv/trace_server.py", "--host", "0.0.0.0", "--port", "8000"]
