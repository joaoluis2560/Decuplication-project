FROM tiangolo/uwsgi-nginx-flask


WORKDIR /deduplicator_final_LSH/

COPY requirements.txt /deduplicator_final_LSH/
RUN pip install -r./requirements.txt


ENV STATIC_PATH /deduplicator_final_LSH/static
COPY   . /deduplicator_final_LSH/

EXPOSE 8080

CMD [ "python", "app.py"]
     