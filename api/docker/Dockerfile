FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy cortex library
COPY cortex ./cortex

# Copy API application
COPY api/app ./app

# Create a non-root user
RUN useradd -m -u 1000 cortex_user && chown -R cortex_user:cortex_user /app
USER cortex_user

EXPOSE 7001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7001"]