# File: Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to reduce image size
# We install torch without CUDA support for a smaller image size.
# For GPU support, you would use a different base image and torch installation.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy the backend script into the container
COPY Backend.py .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for the public key.
# It's recommended to override this at runtime (e.g., with `docker run -e ...`)
ENV NEXA_PUBLIC_KEY="NEXA_PUBLIC_KEY_FOR_DEMO"

# Run Backend.py when the container launches
# Use 0.0.0.0 to allow connections from outside the container
CMD ["uvicorn", "Backend:app", "--host", "0.0.0.0", "--port", "8000"]
