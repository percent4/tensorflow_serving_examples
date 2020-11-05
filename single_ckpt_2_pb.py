# -*- coding: utf-8 -*-
# @Time : 2020/11/5 23:10
# @Author : Jclian91
# @File : single_ckpt_2_pb.py
# @Place : Yangpu, Shanghai
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python import saved_model

export_path = "pb_models/add/1"

graph = tf.Graph()
saver = tf.train.import_meta_graph("./ckpt_models/add/add.ckpt.meta", graph=graph)
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint("./ckpt_models/add"))
    saved_model.simple_save(session=sess,
                            export_dir=export_path,
                            inputs={"t": graph.get_operation_by_name('t').outputs[0]},
                            outputs={"z": graph.get_operation_by_name('z').outputs[0]})
