FROM python:3.11

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install \
		build-essential \
		python3-pip \
		python3-wheel \
		vim \
		git \
		tig \
		tmux \
		graphviz

RUN pip install --upgrade pip

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
