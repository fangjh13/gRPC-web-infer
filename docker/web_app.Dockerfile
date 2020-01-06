FROM python:3.6.10-stretch

WORKDIR /srv

COPY requirements/web.txt requirements.txt
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

COPY . .

# config web connect `inference` service dinfine in docker-compose
RUN sed -i 's/localhost/inference/g' web_app.py

EXPOSE 5000

ENTRYPOINT ["/usr/local/bin/python", "web_app.py"]