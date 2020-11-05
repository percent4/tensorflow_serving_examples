# -*- coding: utf-8 -*-
# @Time : 2020/11/5 23:55
# @Author : Jclian91
# @File : multi_tf_serving.py
# @Place : Yangpu, Shanghai
import requests


# 利用tensorflow/serving的HTTP接口请求进行预测
def model_predict(model_name):
    t = 4.0
    tensor = {"instances": [{"t": t}]}

    url = "http://192.168.1.193:8551/v1/models/{}:predict".format(model_name)
    req = requests.post(url, json=tensor)
    if req.status_code == 200:
        z = req.json()['predictions'][0]
        print("model_{}: ".format(model_name), z)


model_predict("add")
model_predict("subtract")
model_predict("multiply")
model_predict("divide")

'''
model_add:  8.0
model_subtract:  0.0
model_multiply:  16.0
model_divide:  1.0
'''