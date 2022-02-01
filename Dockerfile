FROM python:3

WORKDIR /Haiku-Playground

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /Haiku-Playground/src

CMD [ "jupyter", "notebook" ]