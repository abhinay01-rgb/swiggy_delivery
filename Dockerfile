FROM python:3.12-slim
RUN apt-get update && apt-get install -y libgomp1
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY app.py ./
EXPOSE 5000
CMD [ "python", "./app.py" ]
