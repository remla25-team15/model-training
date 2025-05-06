FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY nltk_data /usr/local/nltk_data
ENV NLTK_DATA=/usr/local/nltk_data

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY datasets/ datasets/
RUN mkdir -p output

ENTRYPOINT ["python"]
CMD ["src/train_model.py", "--dataset", "datasets/a1_RestaurantReviews_HistoricDump.tsv", "--output", "output/"]
