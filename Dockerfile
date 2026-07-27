FROM python:3.12-slim

WORKDIR /app

# Core API dependencies only (see requirements-ml.txt / requirements-dashboard.txt
# for the optional embedding-retrieval and Streamlit-dashboard extras, which are
# not required to run the deployed API and are intentionally left out of the image).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
