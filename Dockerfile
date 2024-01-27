FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# Installing dependencies:
ENV DEBIAN_FRONTEND noninteractive
RUN apt update && \
    apt install -y python3-dev python3-pip && \
    apt install -y python3-opencv && \
    apt install -y git

# Copying requirements and defining workdir:
COPY ./src/requirements.txt /app/src/requirements.txt
WORKDIR /app/src

# Installing python dependencies:
RUN python3 -m pip install -r requirements.txt

# Copying the rest of the files in src
COPY ./src /app/src

# Adding environment variables
ENV MY_MODELS_PATH ${MY_MODELS_PATH}
ENV DATASETS_DIR_PATH ${DATASETS_DIR_PATH}

# CMD ["python3", "main.py"]
CMD ["python3", "benchmarks/pad/pad.py"]