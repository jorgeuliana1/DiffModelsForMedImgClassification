FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# Installing dependencies:
RUN apt update && \
    apt install -y python3-dev python3-pip

# Copying workdir:
COPY ./src /app/src
WORKDIR /app/src

# Installing python dependencies:
RUN python3 -m pip install -r requirements.txt

CMD ["python3", "main.py"]