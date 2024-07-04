FROM  msoo.ces.myfiinet.com:6702/morioka/tiny-openai-whisper-api


# Pip install the dependencies
RUN pip install --upgrade pip 
RUN pip install whisper-timestamped

# Copy the current directory contents into the container at /app
COPY main_timestamped.py /app/main.py

# Set the working directory to /app
WORKDIR /app

# Expose port 8000
EXPOSE 8000

# Run the app
CMD uvicorn main:app --host 0.0.0.0
