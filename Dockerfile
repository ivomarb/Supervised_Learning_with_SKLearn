FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages.
RUN apt-get -y update && \
    apt-get -y install jq awscli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt

ADD . /

#ENTRYPOINT python ./app.py $ARGUMENTS
