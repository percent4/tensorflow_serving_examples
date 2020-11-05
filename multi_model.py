# -*- coding: utf-8 -*-
# @Time : 2020/11/5 23:39
# @Author : Jclian91
# @File : multi_model.py
# @Place : Yangpu, Shanghai
import tensorflow as tf

# add model
with tf.Graph().as_default() as g:
    x = tf.Variable(2.0, dtype=tf.float32, name="x")
    y = tf.Variable(2.0, dtype=tf.float32, name="y")
    xy = x * y
    t = tf.placeholder(shape=None, dtype=tf.float32, name="t")
    z = tf.add(xy, t, name="z")


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(z, feed_dict={t: 3.0})
    print("result: ", result)

    # save the model
    saver = tf.train.Saver()
    saver.save(sess, save_path='./ckpt_models/add/add.ckpt')

# substract model
with tf.Graph().as_default() as g:
    x = tf.Variable(2.0, dtype=tf.float32, name="x")
    y = tf.Variable(2.0, dtype=tf.float32, name="y")
    xy = x * y
    t = tf.placeholder(shape=None, dtype=tf.float32, name="t")
    z = tf.subtract(xy, t, name="z")


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(z, feed_dict={t: 3.0})
    print("result: ", result)

    # save the model
    saver = tf.train.Saver()
    saver.save(sess, save_path='./ckpt_models/subtract/subtract.ckpt')

# multipy
with tf.Graph().as_default() as g:
    x = tf.Variable(2.0, dtype=tf.float32, name="x")
    y = tf.Variable(2.0, dtype=tf.float32, name="y")
    xy = x * y
    t = tf.placeholder(shape=None, dtype=tf.float32, name="t")
    z = tf.multiply(xy, t, name="z")


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(z, feed_dict={t: 3.0})
    print("result: ", result)

    # save the model
    saver = tf.train.Saver()
    saver.save(sess, save_path='./ckpt_models/multiply/multiply.ckpt')

# divide model
with tf.Graph().as_default() as g:
    x = tf.Variable(2.0, dtype=tf.float32, name="x")
    y = tf.Variable(2.0, dtype=tf.float32, name="y")
    xy = x * y
    t = tf.placeholder(shape=None, dtype=tf.float32, name="t")
    z = tf.divide(xy, t, name="z")


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(z, feed_dict={t: 3.0})
    print("result: ", result)

    # save the model
    saver = tf.train.Saver()
    saver.save(sess, save_path='./ckpt_models/divide/divide.ckpt')