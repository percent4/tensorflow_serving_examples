# -*- coding: utf-8 -*-
# @Time : 2020/11/5 23:45
# @Author : Jclian91
# @File : multi_ckpt_2_pb.py
# @Place : Yangpu, Shanghai

import tensorflow as tf
from tensorflow.python import saved_model


# change ckpt file to pb file
def model_export(model_name):
    export_path = "pb_models/{}/1".format(model_name)
    graph = tf.Graph()
    saver = tf.train.import_meta_graph("./ckpt_models/{}/{}.ckpt.meta".format(model_name, model_name),
                                       graph=graph)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, tf.train.latest_checkpoint("./ckpt_models/{}".format(model_name)))
        saved_model.simple_save(session=sess,
                                export_dir=export_path,
                                inputs={"t": graph.get_operation_by_name('t').outputs[0]},
                                outputs={"z": graph.get_operation_by_name('z').outputs[0]})


model_export("add")
model_export("subtract")
model_export("multiply")
model_export("divide")
