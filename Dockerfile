FROM python:3.12-slim-buster

WORKDIR /app

COPY . /app

RUN apt update && apt install awscli -y

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
