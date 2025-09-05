FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "recipe_buddy_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
