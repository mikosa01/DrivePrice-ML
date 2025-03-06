# Use the official Python 3.10 image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the entire backend folder into the container
COPY . /app

# Copy only requirements first to leverage Docker caching
COPY class_api/requirements.txt /app/class_api/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/class_api/requirements.txt

# Expose the application port
EXPOSE $PORT

# Run the application
CMD [ "python", "-m", "class_api.api.app" ]