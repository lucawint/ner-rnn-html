FROM ubuntu:16.04
RUN apt-get update \
    && apt-get upgrade -y

# Set locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Clone repo
WORKDIR /home
RUN apt-get install -y git
RUN git clone https://github.com/lucawint/ner-rnn-html /home/ner

WORKDIR /home/ner

# Install needed libraries
RUN apt-get install -y python3-pip \
    && pip3 install -r requirements.txt \
    && apt-get clean

# Copy trained models & encodings
COPY models/ ./models/

EXPOSE 8000
CMD python3 server.py
