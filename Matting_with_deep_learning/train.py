import VGG
import Generate
import numpy as np
import tools
import tensorflow as tf


learning_rate = 0.0001
global_step = 10
saver_path = "/tmp/deep_matting/model_save/"
saver_file = saver_path + "model"
batch_size = 16
outername = ['F/','B/','I/']
width  = Generate.width
height = Generate.height
is_train = True

F = tf.placeholder(tf.float32,[None, width + 1, height + 1, 3])
B = tf.placeholder(tf.float32,[None, width + 1, height + 1, 3])
I = tf.placeholder(tf.float32,[None, width + 1, height + 1, 3])
alpha_diff = tf.placeholder(tf.float32, [None, 1])




with tf.variable_scope("x1"):
    x1 = tools.FC_layer('fc1',F,out_nodes=1024) 
with tf.variable_scope("x2"):
    x2 = tools.FC_layer('fc1',B,out_nodes=1024)
with tf.variable_scope("x3"):
    x3 = tools.FC_layer('fc1',I,out_nodes=1024)

# x1 = VGG.VGG16N('x', F)
# x2 = VGG.VGG16N('x', B)
# x3 = VGG.VGG16N('x', I)

x = tf.concat([x1, x2, x3], 1)

x = VGG.VGG16N(tf.reshape(x, [-1,32,32,3]) , True)

x = tools.FC_layer('fc9', x, out_nodes=4096)
# with tf.name_scope('batch_norm3'):
#     x = tools.batch_norm(x)           

x = tools.FC_layer('fc10', x, out_nodes=4096)
# with tf.name_scope('batch_norm4'):
#     x = tools.batch_norm(x)           

x = tools.FC_layer('fc11', x, out_nodes=1024)
# with tf.name_scope('batch_norm5'):
#     x = tools.batch_norm(x)           

x = tools.FC_layer('fc12', x, out_nodes=64)
# with tf.name_scope('batch_norm6'):
#     x = tools.batch_norm(x)           

x = tools.FC_layer('fc13', x, out_nodes=1, name="x")
loss = tf.reduce_mean(tf.abs(tf.subtract(x,alpha_diff)), name="loss")
tf.summary.scalar('loss', loss)

# Define the optimizer !
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)


# Define training step !
train_op = optimizer.minimize(loss)

saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)

if is_train:
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./train_log/', sess.graph)
        sess.run(init)
        for v in [n.name for n in tf.get_default_graph().as_graph_def().node]:
            print v
        tools.load_with_skip('/tmp/deep_matting/vgg16.npy', sess, ['fc6', 'fc7', 'fc5', 'fc8'])
        for idx in range(100000):
            batch = Generate.next(batch_size)
            F_train = np.array([x['F'] for x in batch])
            B_train = np.array([x['B'] for x in batch])
            I_train = np.array([x['I'] for x in batch])
            alpha_diff_target = np.array([x['alpha_diff'] for x in batch]).reshape([-1, 1])
            summary, _ = sess.run([merged, train_op], feed_dict={F:F_train, B:B_train, I:I_train, alpha_diff:alpha_diff_target})
            # print 'the idx is %05d'% idx, 'after ',pow(sess.run(loss, feed_dict={F:F_train, B:B_train, I:I_train, alpha_diff:alpha_diff_target}), 0.5)
            writer.add_summary(summary, idx)
            if idx % 1000 == 0:
                learning_rate *= 0.985


            if idx % 200 == 0:
                print 'the idx is %05d'% idx, 'before',sess.run(loss, feed_dict={F:F_train, B:B_train, I:I_train, alpha_diff:alpha_diff_target})
                for v in (zip(alpha_diff_target, sess.run(tf.get_default_graph().get_tensor_by_name("fc13/x:0"),
                    feed_dict={F:F_train, B:B_train, I:I_train, alpha_diff:alpha_diff_target}))):
                    print("%-.20f\t%-.20f\t%-.20f" % (v[0][0] , v[1][0], abs(v[0][0] - v[1][0]))) 
        saver.save(sess, saver_file)
else:
    with tf.Session(config=config) as sess:
        # restore the parameters with path
        saver.restore(sess, tf.train.latest_checkpoint(saver_path))
        batch = Generate.next(batch_size)
        F_train = np.array([x['F'] for x in batch])
        B_train = np.array([x['B'] for x in batch])
        I_train = np.array([x['I'] for x in batch])
        alpha_diff_target = np.array([x['alpha_diff'] for x in batch]).reshape([-1, 1])
        # for v in [n.name for n in tf.get_default_graph().as_graph_def().node]:
        #     print v
        print(sess.run(tf.get_default_graph().get_tensor_by_name("loss/loss:0"), feed_dict={F:F_train, B:B_train, I:I_train, alpha_diff:alpha_diff_target}))
        print(sess.run(tf.get_default_graph().get_tensor_by_name("fc13/x:0"), feed_dict={F:F_train, B:B_train, I:I_train, alpha_diff:alpha_diff_target}))
