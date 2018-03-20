import VGG
import Generate
import numpy as np
import tools
import tensorflow as tf


learning_rate = 0.0001
global_step = 10
batch_size = 128
outername = ['F/','B/','I/']
width  = Generate.width
height = Generate.height

F = tf.placeholder(tf.float32,[None, width + 1, height + 1, 3])
B = tf.placeholder(tf.float32,[None, width + 1, height + 1, 3])
I = tf.placeholder(tf.float32,[None, width + 1, height + 1, 3])
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
    # with tf.name_scope('batch_norm3'):
    #     x = tools.batch_norm(x)           

    tools.FC_layer('Outer/','fc10', x, out_nodes=4096)
    # with tf.name_scope('batch_norm4'):
    #     x = tools.batch_norm(x)           

    tools.FC_layer('Outer/','fc11', x, out_nodes=1024)
    # with tf.name_scope('batch_norm5'):
    #     x = tools.batch_norm(x)           

    # tools.FC_layer('Outer/','fc12', x, out_nodes=64)
    # with tf.name_scope('batch_norm6'):
    #     x = tools.batch_norm(x)           

    tools.FC_layer('Outer/','fc13', x, out_nodes=1)
    return x
    

with tf.name_scope('optimizer'):
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    x = train()
    losss = loss(x)
    train_op = optimizer.minimize(losss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./train', sess.graph)
    sess.run(init)
    tools.load_with_skip(outername, '/tmp/deep_matting/vgg16.npy', sess, ['fc6', 'fc7', 'fc5', 'fc8'])
    tf.summary.scalar('loss', losss)
    for idx in range(100):
        batch = Generate.next(batch_size)
        F_train = np.array([x['F'] for x in batch])
        B_train = np.array([x['B'] for x in batch])
        I_train = np.array([x['I'] for x in batch])
        alpha_diff_target = np.array([x['alpha_diff'] for x in batch]).reshape([-1, 1])
	print 'the idx is %05d'% idx, 'before',sess.run(losss, feed_dict={F:F_train, B:B_train, I:I_train, alpha_diff:alpha_diff_target})
        sess.run(train_op, feed_dict={F:F_train, B:B_train, I:I_train, alpha_diff:alpha_diff_target})
        print 'the idx is %05d'% idx, 'after ',sess.run(losss, feed_dict={F:F_train, B:B_train, I:I_train, alpha_diff:alpha_diff_target})







    
