# TODO 2: Make sure the Dockerfile contains the correct instructions to build and run your AI application.
# Use an official Python runtime as a parent image
# TODO 2: Update the Python version if needed Base Image: Ensure the base image (python:3.9-slim in this example) is suitable for your project.
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
# Expose Port: Verify that the correct port is exposed (if your application is accessible via a network).
EXPOSE 80

# Define the command to run the model script (update with your script name)
CMD ["python", "./demo.py"]


# Before pushing the workflow to GitHub, it’s a good practice to build and test the Docker image locally.

# Build the Docker Image Locally:

# Open a terminal window.
# docker build -t my-ai-agent .
# . specifies the build context, which is typically the directory containing your Dockerfile.

# Run the Docker Image Locally:
# docker run -p 8000:8000 my-ai-agent
# This command runs the Docker container and maps port 8000 of the container to port 8000 on your local machine (adjust the port if your application uses a different one).
# Check the application by accessing http://localhost:8000 in your browser or via a tool like curl

# Test Application in the Container:

#     Verify that the application performs as expected (e.g., outputs correct responses or loads without errors).
#     Run any additional commands in the container to verify functionality, like testing endpoints or running scripts.
#     Stop the Docker Container:

    #docker stop <container-id>
    # Replace <container-id> with the ID of the running container (use docker ps to find it).


    # Step 3: Push the Workflow File to GitHub
    # Once you have verified that the Dockerfile works correctly and your application runs smoothly in a container, push the workflow file to GitHub to trigger the automated CI/CD pipeline.
    
    # Commit and Push the Workflow File:
    
    # bash
    # Copy code
    # git add .github/workflows/ci.yml
    # git commit -m "Add CI workflow for Docker build and run"
    # git push origin main
    # Monitor GitHub Actions Workflow:
    
    # Go to the Actions tab in your GitHub repository.
    # You’ll see the workflow triggered by the push event.
    # Click on the workflow run to view logs for each step (build, test, deploy).
    # Debugging:
    
    # If any step fails, check the logs under the corresponding job to identify and resolve issues.
    # Common errors could be missing dependencies, incorrect Dockerfile instructions, or issues with the application code.