FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Copy dependency list first for better cache use
COPY requirements.txt /app/requirements.txt

# Ensure pip is recent
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python deps. Use --prefer-binary to favor wheels and reduce compile-time
# memory pressure when possible. Note: some packages (torch) may still be large.
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose backend and streamlit ports
EXPOSE 8000 8501

# Add a lightweight start script and make sure it will run via bash
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["bash", "/start.sh"]
