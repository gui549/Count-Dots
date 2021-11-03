FROM python:3.8-buster

RUN set -ex && mkdir /repo
WORKDIR /repo

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip~=21.0.0
RUN pip install -r requirements.txt

COPY dataloader.py ./dataloader.py
COPY models.py ./models.py
COPY utils.py ./utils.py
COPY train.py ./train.py
COPY test.py ./test.py

ENV PYTHONPATH /repo
ENTRYPOINT ["python3", "./train.py"]

--root ./datasets/DotsEven/ -f base -l -m resnet_scalar --batch_size 32
