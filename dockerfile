# Python base image  
FROM python:3.6.3  
  
# Where the API server lives  
WORKDIR /app/  
  
# Install required dependencies  
COPY requirements.txt /app/  
RUN pip install -r ./requirements.txt  
  
# API server  
ADD app /app/
# Container port on which the server will be listening  
EXPOSE 7777  
# Launch server app  
ENTRYPOINT python ./score.py  