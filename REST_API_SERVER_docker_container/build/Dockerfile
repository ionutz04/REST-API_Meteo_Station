# Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir flask redis

# Expose the Flask port
EXPOSE 5000

# Run the application
CMD ["python", "main.py"]
