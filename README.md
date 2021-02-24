本项目用于演示如何使用tensorflow/serving进行单模型与多模型部署以及模型预测。

拉取tensorflow/serving镜像：

```
docker pull tensorflow/serving:1.14.0
```

单模型部署命令：

```
docker run -t --rm -p 8551:8501 -v "absolute_path_to_pb_models/pb_models/add:/models/add" -e MODEL_NAME=add tensorflow/serving:1.14.0
```

多模型部署命令：

```
docker run -t -d --rm -p 8551:8501 -v "absolute_path_to_pb_models/pb_models:/models" tensorflow/serving:1.14.0 --model_config_file=/models/models.config
```

模型多版本部署命令：

```
docker run -t -d --rm -p 8551:8501 -v "absolute_path_to_pb_models/pb_models:/models" tensorflow/serving:1.14.0 --model_config_file=/models/models.config
```

模型预测：

参考`single_tf_serving.py`及`multi_tf_serving.py`、`version_control_tf_serving.py`。

curl命令：

```
curl --location --request POST 'http://192.168.1.193:8551/v1/models/add/versions/2:predict' \
--header 'Content-Type: application/json' \
--data-raw '{
    "instances": [{"t": 2}]
}'
```

输出：

```
{
    "predictions": [
        8.0
    ]
}
```

### 使用Prometheus进行服务监控

1. Prothemus的配置文件位于./pb_models/prometheus.yml; Tensorflow/Serving支持Prothemus的配置文件位于pb_models/monitoring_config.txt;
2. 启动服务：

```
docker-compose up -d
```

3. 在浏览器中输入http://IP:9090, 选择Graph, 配置好Expression（比如:tensorflow:core:graph_run_time_usecs_histogram_count，调用次数）即可使用。