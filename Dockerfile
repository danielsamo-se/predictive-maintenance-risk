FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --upgrade pip && \
    pip install . && \
    pip install fastapi "uvicorn[standard]"

COPY configs/ ./configs/

EXPOSE 8000

CMD ["uvicorn", "pmrisk.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
