#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
import Train_data as train_data

import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
    前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
    input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
    filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
    strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
    padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
    use_cudnn_on_gpu 是否使用cudnn加速。默认是True
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
    tf.nn.max_pool 进行最大值池化操作,而avg_pool 则进行平均值池化操作
    几个参数分别是：value, ksize, strides, padding,
    value:  一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样
    ksize:  长为4的list,表示池化窗口的尺寸
    strides: 窗口的滑动值，与conv2d中的一样
    padding: 与conv2d中用法一样。
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 20, 20, 9])
x_image = tf.reshape(x, [-1,20,20,9]) #将输入按照 conv2d中input的格式来reshape，reshape


"""
# 第一层
# 卷积核(filter)的尺寸是5*5, 通道数为1，输出通道为32，即feature map 数目为32
# 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*28*28*32
# 也就是单个通道输出为28*28，共有32个通道,共有?个批次
# 在池化阶段，ksize=[1,2,2,1] 那么卷积结果经过池化以后的结果，其尺寸应该是？*14*14*32
"""
W_conv1 = weight_variable([5, 5, 9, 32])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"""
# 第二层
# 卷积核5*5，输入通道为32，输出通道为64。
# 卷积前图像的尺寸为 ?*14*14*32， 卷积后为?*14*14*64
# 池化后，输出的图像尺寸为?*7*7*64
"""
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三层 是个全连接层,输入维数7*7*64, 输出维数为1024
W_fc1 = weight_variable([5 * 5 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*64])
h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32) # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第四层，输入1024维，输出50维
W_fc2 = weight_variable([1024, 50])
b_fc2 = bias_variable([50])
h_pool2_flat = tf.reshape(h_fc1_drop, [-1, 1024])
h_fc2 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)



# 第五层，输入50维，输出1维
W_fc3 = weight_variable([50, 1])
b_fc3 = bias_variable([1])
h_pool3_flat = tf.reshape(h_fc2, [-1, 50])
h_fc3 = tf.nn.elu(tf.matmul(h_pool3_flat, W_fc3) + b_fc3)
y_ = tf.placeholder(tf.float32, [None, 1])


h_fc4 = tf.sigmoid(h_fc3)

cross_entropy = tf.reduce_mean(tf.abs(y_ - h_fc4))
# cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=h_fc3))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(h_fc3), reduction_indices=[1])) # 损失函数，交叉熵
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy) # 使用GradientDescentOptimizer优化
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # 计算准确度
correct_prediction = tf.abs(y_-h_fc4)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(tf.initialize_all_variables()) # 变量初始化
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = train_data.train_data.next_batch(1)
    sess.run(train_step, feed_dict={x: batch[0]/255, y_: batch[1], keep_prob: 0.5})
    if i%10 == 0:
        batch = train_data.train_data.next_batch(100)
        train_accuracy = sess.run(accuracy, feed_dict={x:batch[0]/255, y_: batch[1]/255, keep_prob: 1.0})
        c = sess.run(cross_entropy, feed_dict={x: batch[0]/255, y_: batch[1], keep_prob: 0.5})
        print(sess.run(h_fc4, feed_dict={x: batch[0]/255, y_: batch[1], keep_prob: 0.5}))
        print("step %d, training accuracy %.10f, loss is %f"%(i, train_accuracy, c))

# print("test accuracy %.10f"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
