# docker常用命令

|Index	|命令项	|解释| 使用场景|
| ---   | -----| --- | ---|
|1	|docker pull 	|从仓库拉取所需要的镜像 | 网络下载镜像|
|2	|docker images	|查看本地已有的镜像（下载的或自己构建的）|
|3	|docker commit	|提交镜像|自己修改了镜像后，提交到本地，供后续直接使用|
|4	|docker save/load	|分别表示导出镜像到本地或加载由save保存的本地文件镜像|镜像保存为文件，然后拷贝到其他主机|
|5	|docker push	|上传到仓库中来共享（对于上传到Docker Hub需要提前注册）|将自己的镜像分享到网络上，供大家使用|
|6	|docker rmi	|删除镜像 |
|7	|docker run	|运行镜像即启动容器 |后面接容器名或ID，能够唯一标示容器的代号|
|8	|docker start/stop	|启动已停止运行的容器/终止一个运行中的容器|
|9	|docker ps	|查看正在运行的容器| 加参数 -a 可以看到已经停止运行的容器|
|10	|docker attach	|进入容器 | 容器运行后，进入到里面|
|11	|docker export/import	|导出和导入容器 | 与save/load不同，export/import不再保留构建的层级信息|
|12	|docker rm	|删除已经停止的容器（删除容器前，先docker stop）|


**操作示例**

1、拉取python3.6的镜像： 

```python
docker pull python:3.6-alpine 
```

2、查看镜像：

```python
#  当前显示镜像ID 83d065b0546b
docker images
```

3、运行并进入镜像：

```python
docker run --name py36_docker -i -t 83d065b0546b  /bin/sh
```
4、在容器内运行Python

```python
Python 3.6.8 (default, Feb  6 2019, 01:56:13) 
[GCC 8.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

此后，我们可以在该镜像里做相关操作（如某个项目的数据分析）而不影响外部的环境，加入-v的参数，将本地路径a映射到容器中的镜像b:

```python
docker run  -v /a:/b 
```

这样就可以将数据文件放入目录a，却在容器的目录b中进行分析，后续分析产生的结果同样会在a中出现，实现了容器快速安装某个软件的方法。

# 数据科学项目标准化

下载本书提供的镜像：https://hub.docker.com
搜索： chansonz/ml_dev_env

## 新建项目工程

```
[ ~ ]# cookiecutter  ~/cute-datascience-sem
project_name [project_name]: example_project
repo_name [example_project]: 
author_name [Your name (or your organization/company/team)]: chansonz
description [A short description of the project.]: example for readers
Select open_source_license:
1 - MIT
2 - BSD-3-Clause
3 - No license file
Choose from 1, 2, 3 [1]: 
Select python_interpreter:
1 - python3
2 - python
Choose from 1, 2 [1]: 
Select use_nvidia_docker:
1 - no
2 - yes
Choose from 1, 2 [1]: 1
```

执行上述的命令后，在当前目录生成example_project文件夹。其目录结构如下：

```
[ example_project ]# tree
.
├── data
│   ├── external
│   ├── processed
│   ├── raw
│   └── tmp
├── docker-compose_with_build.yml
├── docker-compose.yml
├── Dockerfile
├── docs
│   ├── commands.rst
│   ├── conf.py
│   ├── getting-started.rst
│   ├── index.rst
│   ├── make.bat
│   └── Makefile
├── LICENSE
├── Makefile
├── models
├── notebooks
├── README.md
├── references
├── reports
│   └── figures
├── requirements.txt
├── src
│   ├── data
│   │   └── make_dataset.py
│   ├── features
│   │   └── build_features.py
│   ├── __init__.py
│   ├── models
│   │   ├── predict_model.py
│   │   └── train_model.py
│   └── visualization
│       └── visualize.py
├── start.sh
├── stop.sh
├── test_environment.py
└── tox.ini
16 directories, 23 files
```

## 管理工具

|Index	|命令项|	解释                                   |
|----| -----| ----|
|1	|clean	|删除所有已编译的Python文件,即pyc和cache等文件         |
|2	|clean-container	|停止容器并删除                      |
|3	|clean-data	|删除data文件下的所有文件                   |
|4	|clean-docker	|删除容器和镜像，慎用                     |
|5	|clean-image	|删除镜像                             |
|6	|clean-model	|删除model文件夹下内容                    |
|7	|create_environment	|在本地创建环境，不是在docker中创建     |
|8	|data	|使用本地Python调用src/data/make_dataset.py   |
|9	|lint	|使用本地flake8静态检查工具对src文件夹下的代码规范进行检查      |
|10	|profile	|查看当前容器，镜像和已开启的端口信息                   |
|11	|requirements	|在本地环境中安装flake8等包               |
|12	|test_environment	|只检查了Python版本，未做其他检查，读者可自行完善|



