# -*- coding: utf-8 -*-
# @Time : 2020/11/6 9:50
# @Author : Jclian91
# @File : version_control_ckpt_2_pb.py
# @Place : Yangpu, Shanghai
import tensorflow as tf
from tensorflow.python import saved_model


# change ckpt file to pb file
def model_export(model_version, tf_version):
    export_path = "pb_models/add/{}".format(tf_version)
    graph = tf.Graph()
    saver = tf.train.import_meta_graph("./ckpt_models/add/{}.ckpt.meta".format(model_version),
                                       graph=graph)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, tf.train.latest_checkpoint("./ckpt_models/add"))
        saved_model.simple_save(session=sess,
                                export_dir=export_path,
                                inputs={"t": graph.get_operation_by_name('t').outputs[0]},
                                outputs={"z": graph.get_operation_by_name('z').outputs[0]})


model_export("add", 1)
model_export("add2", 2)
model_export("add3", 3)