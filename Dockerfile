FROM docker.io/python:3.8

ARG HOST_UID
ARG HOST_GID
ARG HOST_USER

#Create a non-root user to match the user that created this docker container
RUN groupadd --gid $HOST_GID $HOST_USER && \
    useradd --no-log-init --create-home --shell /bin/bash \
    --uid $HOST_UID --gid $HOST_GID $HOST_USER

RUN apt update
#Note research was performed with imagemagick 6.9.7-4
RUN apt install -y imagemagick 

# Install dependencies from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY . /app
WORKDIR /app
RUN pip install -e .

#Set a non-root user
USER $HOST_UID

CMD jupyter lab --port=8888 --ip=0.0.0.0 --no-browser \
    --notebook-dir=/app/image_compression_attribution/common/publications/2021-summer-attrib
EXPOSE 8888
