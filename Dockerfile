# Copy the requirements file into the container
FROM python:3.9.7

# Set the working directory in the container
WORKDIR /app

# Install the dependencies specified in the requirements file
COPY requirements.txt requirements.txt

# Install the dependencies specified in the requirements file
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 5001 to allow the Flask app to be accessible
EXPOSE 5001

# Define the command to run the application
CMD ["python", "app.py"]
