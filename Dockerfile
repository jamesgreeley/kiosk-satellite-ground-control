# CPU image: the Home Assistant app and standalone Docker (amd64 and arm64).
#   docker build -t kiosk-satellite-ground-control .
FROM python:3.12-slim

ARG BUILD_VERSION=dev
ARG BUILD_ARCH=""
LABEL org.opencontainers.image.title="Kiosk Satellite Ground Control" \
      org.opencontainers.image.source="https://github.com/jxlarrea/kiosk-satellite-ground-control" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      io.hass.type="addon" io.hass.name="Kiosk Satellite Ground Control" \
      io.hass.version="${BUILD_VERSION}" io.hass.arch="${BUILD_ARCH}"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODELS_DIR=/app/models DATA_DIR=/data

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY ground_control ./ground_control
RUN pip install --no-cache-dir . && rm -rf /root/.cache
COPY models/silero_vad.onnx models/redimnet2_b3_multi.onnx models/redimnet2_b1.onnx models/voice_isolator_cpu.onnx \
     models/*.LICENSE.md ./models/

VOLUME /data
EXPOSE 10900
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"10900\")}/health')"
CMD ["python", "-m", "ground_control"]
