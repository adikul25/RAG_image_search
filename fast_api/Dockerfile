FROM python:3.13.0-slim

RUN mkdir /FastAPI

COPY requirements.txt /FastAPI

WORKDIR /FastAPI

RUN apt-get update -y

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /FastAPI/app

COPY ./.env /FastAPI/app/.env

COPY ./app/image_rag /FastAPI/app/image_rag

RUN chmod -R 755 /FastAPI/app/image_rag

EXPOSE 8000

WORKDIR /FastAPI/app

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]
