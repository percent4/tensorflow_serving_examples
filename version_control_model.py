# -*- coding: utf-8 -*-
# @Time : 2020/11/6 9:48
# @Author : Jclian91
# @File : version_control_model.py
# @Place : Yangpu, Shanghai
import tensorflow as tf

# 第一个模型
g = tf.Graph()
with g.as_default() as g:
    x = tf.Variable(2.0, dtype=tf.float32, name="x")
    y = tf.Variable(2.0, dtype=tf.float32, name="y")
    xy = x * y
    t = tf.placeholder(shape=None, dtype=tf.float32, name="t")
    z = tf.add(xy, t, name="z")


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(z, feed_dict={t: 1.0})
    print("result: ", result)

    # save the model
    saver = tf.train.Saver()
    saver.save(sess, save_path='./ckpt_models/add/add.ckpt')


# 第二个模型
g = tf.Graph()
with g.as_default() as g:
    x = tf.Variable(2.0, dtype=tf.float32, name="x")
    y = tf.Variable(2.0, dtype=tf.float32, name="y")
    xy = x * y
    t = tf.placeholder(shape=None, dtype=tf.float32, name="t")
    z = tf.add(xy, 2*t, name="z")


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(z, feed_dict={t: 1.0})
    print("result: ", result)

    # save the model
    saver = tf.train.Saver()
    saver.save(sess, save_path='./ckpt_models/add/add2.ckpt')


# 第三个模型
g = tf.Graph()
with g.as_default() as g:
    x = tf.Variable(2.0, dtype=tf.float32, name="x")
    y = tf.Variable(2.0, dtype=tf.float32, name="y")
    xy = x * y
    t = tf.placeholder(shape=None, dtype=tf.float32, name="t")
    z = tf.add(xy, 3*t, name="z")


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(z, feed_dict={t: 1.0})
    print("result: ", result)

    # save the model
    saver = tf.train.Saver()
    saver.save(sess, save_path='./ckpt_models/add/add3.ckpt')
