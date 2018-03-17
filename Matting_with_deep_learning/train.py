import VGG
import Generate
import numpy as np
import tools
import tensorflow as tf


learning_rate = 0.001
global_step = 100
batch_size = 128
outername = ['F/','B/','I/']

F = tf.placeholder(tf.float32,[None, 20, 20, 3])
B = tf.placeholder(tf.float32,[None, 20, 20, 3])
I = tf.placeholder(tf.float32,[None, 20, 20, 3])
alpha_diff = tf.placeholder(tf.float32, [None, 1])

def loss(x):
    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.abs(x-alpha_diff), name='loss')
        tf.summary.scalar(scope+'/loss', loss)
        return loss


def train():
    x1 = VGG.VGG16N('F', F)
    x2 = VGG.VGG16N('B', B)
    x3 = VGG.VGG16N('I', I)

    x = tf.concat([x1, x2, x3], 1)

    tools.FC_layer('Outer/','fc9', x, out_nodes=4096)
    with tf.name_scope('batch_norm3'):
        x = tools.batch_norm(x)           

    tools.FC_layer('Outer/','fc10', x, out_nodes=4096)
    with tf.name_scope('batch_norm4'):
        x = tools.batch_norm(x)           

    tools.FC_layer('Outer/','fc11', x, out_nodes=1024)
    with tf.name_scope('batch_norm5'):
        x = tools.batch_norm(x)           

    tools.FC_layer('Outer/','fc12', x, out_nodes=64)
    with tf.name_scope('batch_norm6'):
        x = tools.batch_norm(x)           

    tools.FC_layer('Outer/','fc13', x, out_nodes=1)
    # with tf.name_scope('batch_norm7'):
    #     x = tools.batch_norm(x)           
    return x
    

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    x = train()
    losss = loss(x)
    train_op = optimizer.minimize(losss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    tools.load_with_skip(outername, '/tmp/deep_matting/vgg16.npy', sess, ['fc6', 'fc7', 'fc5', 'fc8'])
    for idx in range(100000):
        batch = Generate.next(batch_size)
        F_train = np.array([x['F'] for x in batch])
        B_train = np.array([x['B'] for x in batch])
        I_train = np.array([x['I'] for x in batch])
        alpha_diff_target = np.array([x['alpha_diff'] for x in batch]).reshape([batch_size, 1])
        print 'before',sess.run(losss, feed_dict={F:F_train, B:B_train, I:I_train, alpha_diff:alpha_diff_target})
        sess.run(train_op, feed_dict={F:F_train, B:B_train, I:I_train, alpha_diff:alpha_diff_target})
        print 'after ',sess.run(losss, feed_dict={F:F_train, B:B_train, I:I_train, alpha_diff:alpha_diff_target})







    
