# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# Leaving a  comment to test
FROM python:3.11-slim
ENV PYTHONUNBUFFERED True
ENV DEBIAN_FRONTEND noninteractive
# ENV http_proxy http://thd-svr-proxy-qa.homedepot.com:7070
# ENV https_proxy http://thd-svr-proxy-qa.homedepot.com:7070
RUN mkdir /app
ADD . /app
# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN apt-get update
# RUN apt install -y libgl1-mesa-glx
# RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ENV GOOGLE_APPLICATION_CREDENTIALS gap.json
# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt
# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app

# RUN unset http_proxy https_proxy
# ENV http_proxy=
# ENV https_proxy=

EXPOSE 8080
EXPOSE 8082

CMD ["python3", "app.py"]