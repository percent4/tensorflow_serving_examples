本项目用于演示如何使用tensorflow/serving进行单模型与多模型部署以及模型预测。

拉去tensorflow/serving镜像：

```
docker pull tensorflow/serving:1.14.0
```

单模型部署命令：

```
docker run -t --rm -p 8551:8501 -v "absolute_path_to_pb_models/pb_models/add:/models/add" -e MODEL_NAME=add tensorflow/serving
```

多模型部署命令：

```
docker run -t -d --rm -p 8551:8501 -v "absolute_path_to_pb_models/pb_models:/models" tensorflow/serving --model_config_file=/models/models.config
```

模型预测：

参考`single_tf_serving.py`及`multi_tf_serving.py`。