
# mysql_export.sql

```
SELECT * INTO OUTFILE '/tmp/export_mysql.csv' COLUMNS TERMINATED BY ',' LINES TERMINATED BY '\r\n' FROM information_schema.CHARACTER_SETS;
```


# mongo export.sh

```
#!/bin/bash
HOST=xxx
PORT=xxx
USER=xxx
PASS='xxx'
DB_NAME=xxx
TB_NAME=xxx
CSV_FILE=$1
FIELDS_FILE=filelds.txt
CONDITION='{ $and: [{"timestamp": { $gte: 1555516800 } }, {"timestamp": { $lt: 1556207999 } } ] }'
/usr/bin/mongoexport -h ${HOST} --port ${PORT}\
          -d ${DB_NAME} -c ${TB_NAME}         \
          -u ${USER} -p ${PASS}               \
          --readPreference=nearest            \
          --type=csv -q "${CONDITION}"        \
          --fieldFile=$FIELDS_FILE -o $CSV_FILE
```

# spark_export.py

```
from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext
conf = SparkConf().setAppName("pyspark_export_csv")
sc = SparkContext(conf=conf)
hiveCtx = HiveContext(sc)

sql = '''select * from your_db.your_table'''
spk_df = hiveCtx.sql(sql)
df = spk_df.toPandas()
df.to_csv('spark_exported.csv', encoding='utf-8')
```

# hive export.sql

```
insert overwrite local directory '/tmp/hive_export'
row format delimited
fields terminated by ',' # 默认分隔符为8进制1 \x01
select * from your_db.your_table;
```

# impala  your.sql

```
impala-shell -i your_impala_node -f your.sql -B --output_delimiter="," --print_header -o your_data.csv
```


