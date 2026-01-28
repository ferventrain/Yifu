# Use micromamba base image
FROM mambaorg/micromamba:latest

# Switch to root user for all operations
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the environment configuration
# No need for chown since we are root
COPY environment.yml /tmp/env.yaml

# Install dependencies using micromamba
# We install into the 'yifu' environment
# Note: micromamba is in /bin/micromamba, so it's available to root
RUN micromamba create -y -f /tmp/env.yaml && \
    micromamba clean --all --yes

# Add the new environment to PATH directly
ENV PATH /opt/conda/envs/yifu/bin:$PATH

# Copy the test script
COPY test_env.py .

# Define the default command
CMD ["python", "test_env.py"]
