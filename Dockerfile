# Use a lightweight Python image with version >=3.9
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files from the current directory into the container
COPY . .

# Install necessary Python libraries
RUN pip install --no-cache-dir flask scikit-learn==1.6.0 pandas joblib

# Expose the port the Flask app will run on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]
