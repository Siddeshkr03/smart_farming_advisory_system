# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that your Flask app will run on
EXPOSE 8080

# Define environment variable
ENV PORT 8080

# Run the Flask application using Gunicorn
# Replace 'your_app_file' with the name of your main Flask application file (e.g., app.py, main.py)
# Replace 'app' with the name of your Flask application instance (e.g., if you have app = Flask(__name__), use 'app')
CMD ["gunicorn", "--bind", "0.0.0.0:$(PORT)", "your_app_file:app"]