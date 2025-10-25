# Use official slim Python image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY src/ ./src/
COPY examples/ ./examples/

# Add src to PYTHONPATH so imports work
ENV PYTHONPATH=/app/src:${PYTHONPATH:-}

# Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# Default command: run the optimizer
ENTRYPOINT ["python", "-m", "grid_feedback_optimizer.main"]
