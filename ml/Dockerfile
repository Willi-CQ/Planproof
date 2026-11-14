FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=4
WORKDIR /workspace/planproof
COPY ml/requirements.lock /tmp/planproof-requirements.lock
RUN python -m pip install --no-cache-dir wheel==0.45.1 -r /tmp/planproof-requirements.lock
COPY ml /workspace/planproof/ml
COPY data/gpu-training-datasets.csv /workspace/planproof/data/gpu-training-datasets.csv
RUN python -m pip install --no-build-isolation --no-deps -e /workspace/planproof/ml
ENTRYPOINT ["planproof-ml"]
CMD ["doctor"]
