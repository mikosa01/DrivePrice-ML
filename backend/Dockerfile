# Use the official Python 3.10 image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the entire backend folder into the container
COPY . /app

# Copy only requirements.txt first to leverage Docker caching for faster builds
COPY class_api/requirements.txt /app/class_api/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/class_api/requirements.txt

# Expose the application port
EXPOSE $PORT

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "class_api.api.app:create_app()"]


