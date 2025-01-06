FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir kamiwaza

# Copy the rest of the application
COPY . .
RUN pip install -e .

# Final stage
FROM python:3.12-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Set environment variables
ENV OPENAI_API_TYPE=kamiwaza
ENV OPENAI_API_KEY=na
ENV YDC_API_KEY=na

# Set default command
ENTRYPOINT ["python", "examples/storm_examples/run_storm_wiki_gpt.py"]
CMD ["--help"]
