FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# Installing dependencies:
RUN apt update && \
    apt install -y python3-dev python3-pip && \
    apt install -y git

# Copying requirements and defining workdir:
COPY ./src/requirements.txt /app/src/requirements.txt
WORKDIR /app/src

# Installing python dependencies:
RUN python3 -m pip install -r requirements.txt

# Installing raug
RUN python3 -m pip install git+https://github.com/paaatcha/raug.git

# Copying the rest of the files in src
COPY ./src /app/src

CMD ["python3", "main.py"]