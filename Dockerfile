FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Uses server.app:main — matches pyproject.toml and openenv.yaml entry_point
CMD ["python", "-c", "from server.app import main; main()"]
