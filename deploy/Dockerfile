# Reuse the verified CUDA/model runtime on the existing inference server.
ARG INFERENCE_IMAGE=planproof-inference:e761b84
FROM ${INFERENCE_IMAGE}
WORKDIR /workspace/planproof
COPY backend ./backend
COPY feedback ./feedback
COPY deterministic ./deterministic
COPY ml ./ml
RUN python -m pip install --no-cache-dir -r backend/requirements.txt \
    -e ./feedback -e ./deterministic -e ./ml
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/planproof/feedback/src \
    PLANPROOF_ML_PYTHON=/opt/conda/bin/python \
    PLANPROOF_DETERMINISTIC_PYTHON=/opt/conda/bin/python \
    PLANPROOF_ML_DEVICE=cuda \
    PLANPROOF_ML_CHECKPOINT=/models/corrected-3500-global/detector.pt \
    PLANPROOF_ML_CALIBRATION=/models/corrected-3500-global/calibration.json \
    PLANPROOF_ENV_FILE=/run/secrets/providers.env \
    PLANPROOF_GEMINI_ENV_FILE=/run/secrets/providers.env
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/agent', timeout=4)"
# One worker is intentional: canonical pages and the run executor are in memory.
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--app-dir", "backend"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
