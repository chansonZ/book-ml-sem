FROM python:2.7-alpine3.9
EXPOSE 5000
RUN pip install pipenv
RUN mkdir -p /app/src
WORKDIR /app
ADD Pipfile /app
ADD Pipfile.lock /app
RUN pipenv install --system --deploy --ignore-pipfile
ADD src/index.py /app/src
CMD ["python", "/app/src/index.py"]