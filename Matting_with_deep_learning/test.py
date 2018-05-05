import tensorflow as tf
import os
import cv2
import numpy as np
import pdb

F = tf.placeholder(tf.float32, shape=[1, 3])
B = tf.placeholder(tf.float32, shape=[1, 3])
I = tf.placeholder(tf.float32, shape=[1, 3])
cal_alpha = tf.placeholder(tf.float32)

fbi = tf.concat([F, B, I], axis=0) / 255.0
W1 = tf.Variable(tf.truncated_normal([3,1]))
b1 = tf.Variable(tf.truncated_normal([1]))
fb = W1 * fbi + b1
W2 = tf.Variable(tf.truncated_normal([3,1]))
b2 = tf.Variable(tf.truncated_normal([1]))
ib = W2 * fbi + b2
t1 = tf.reduce_sum(fb*ib)
t2 = tf.reduce_sum(fb*fb)
t = t1 / (t2+0.01)
t = tf.where(tf.greater(t1,tf.constant(1.0)),tf.constant(1.0),t)
t = tf.where(tf.less(t1,tf.constant(0.0)),tf.constant(0.0),t)

loss = abs(cal_alpha - t)

train_op = tf.train.AdadeltaOptimizer(learning_rate=1e-8).minimize(loss)

os.environ['CUDA_VISIABLE_DEVICE'] = '0'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for idx in range(1000000):
        batch = np.load("/disk3/Graduate-design/t/{:0>6}.npy".format(idx))[0]
        F_ = batch['F']
        B_ = batch['B']
        I_ = batch['I']
        cal_alpha_ = batch['cal_alpha']
        feed = {F:F_, B:B_, I:I_, cal_alpha:cal_alpha_}
        _,los = sess.run([train_op,loss],feed_dict=feed)
        print los
