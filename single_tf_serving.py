# -*- coding: utf-8 -*-
# @Time : 2020/11/5 23:36
# @Author : Jclian91
# @File : single_tf_serving.py
# @Place : Yangpu, Shanghai
import requests

# 利用tensorflow/serving的HTTP接口请求进行预测
t = 1.0
tensor = {"instances": [{"t": t}]}

url = "http://192.168.1.193:8551/v1/models/add:predict"
req = requests.post(url, json=tensor)
if req.status_code == 200:
    z = req.json()['predictions'][0]
    print("model_add:", z)
