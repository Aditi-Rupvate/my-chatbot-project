# Use an official Python 3.10 image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies:
# - build-essential: The C compiler (gcc) and other tools needed to build software.
# - portaudio19-dev: The specific library required by PyAudio.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python packages from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (api_rag.py, the kb folder, etc.)
COPY . .

# Tell Render that the service will listen on port 10000
EXPOSE 10000

# The command to start your backend server
CMD ["uvicorn", "api_rag:app", "--host", "0.0.0.0", "--port", "10000"]
