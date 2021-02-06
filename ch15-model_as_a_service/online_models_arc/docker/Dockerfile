FROM docker.io/continuumio/anaconda3

# 创建目录，镜像内框架源码目录
RUN mkdir /root/modelserver/  
WORKDIR   /root/modelserver/

COPY ./arc_src/_dataprocess.py  /root/modelserver/dataprocess.py
COPY ./arc_src/initialize.py    /root/modelserver/initialize.py.py
COPY ./arc_src/_scorefromat.py  /root/modelserver/scorefromat.py
COPY ./arc_src/server.py        /root/modelserver/server.py
COPY ./arc_src/utils.py         /root/modelserver/utils.py


# 框架软件包需求 flask
RUN pip install --upgrade pip
COPY ./requirements.txt /root/modelserver/requirements.txt
RUN pip install -r /root/modelserver/requirements.txt

# 用户源码目录，请参考源码 /mnt/src，/mnt/resource，/mnt/log

# docker build -f Dockerfile -t modelserver/py3:v1.0 .