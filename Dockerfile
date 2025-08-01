FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nba_ranker_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "nba_ranker_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
