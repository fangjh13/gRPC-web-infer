FROM tensorflow/tensorflow:2.0.0-py3

WORKDIR /srv

RUN sed -E -i 's/(archive|security)\.(ubuntu|canonical)\.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

COPY requirements/infer.txt requirements.txt
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

COPY . .

EXPOSE 50051

ENTRYPOINT ["/usr/bin/python3", "inference_server.py"]