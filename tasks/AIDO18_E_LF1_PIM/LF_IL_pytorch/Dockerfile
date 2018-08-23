# TODO: Make a RPi compatible image
FROM pytorch/pytorch

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends git && \
    pip install -e git+https://github.com/duckietown/duckietown-slimremote.git#egg=duckietown-slimremote && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . .

RUN pip install -e .
