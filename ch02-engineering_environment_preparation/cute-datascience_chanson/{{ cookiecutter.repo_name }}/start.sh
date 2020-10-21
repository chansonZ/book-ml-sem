#! /bin/bash
docker-compose up -d

dirname=$(basename `pwd`)
ipp=`ifconfig eth0|grep "inet"|awk -F" " '{print $2}'`
port=`docker ps | grep $dirname |awk '{for(i=1;i<=NF;i++) print $i}' | grep "tcp"| awk -F':' '{print $2}'|awk -F'->' '{print $1}'`
echo
echo http://$ipp:$port
echo
