FROM python:3.10

EXPOSE 5000

RUN apt-get update && \
    apt-get install -y \
    libxmlsec1-dev \
    libxmlsec1-openssl \
    tzdata && \ 
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

WORKDIR /work
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt