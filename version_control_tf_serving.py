# -*- coding: utf-8 -*-
# @Time : 2020/11/6 9:58
# @Author : Jclian91
# @File : version_control_tf_serving.py
# @Place : Yangpu, Shanghai
import requests


# 利用tensorflow/serving的HTTP接口请求进行预测
def model_predict(model_version):
    t = 4.0
    tensor = {"instances": [{"t": t}]}

    url = "http://192.168.1.193:8551/v1/models/add/versions/{}:predict".format(model_version)
    req = requests.post(url, json=tensor)
    if req.status_code == 200:
        z = req.json()['predictions'][0]
        print("model_version{}: ".format(model_version), z)


model_predict("1")
model_predict("2")
model_predict("3")
