#!/usr/bin/python
import os
import pdb
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
os.environ['CUDA_VISIBLE_DEVICES']='0'

image_size = 320
train_batch_size = 10

# choose weather to load data to sess
pretrained_model = True
is_train = True

en_parameters = []
pool_parameters = []



def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
    with tf.variable_scope(scope):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]
        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret



F = tf.placeholder(tf.int32, shape = (train_batch_size, image_size, image_size, 3))
B = tf.placeholder(tf.int32, shape = (train_batch_size, image_size, image_size, 3))
I = tf.placeholder(tf.int32, shape = (train_batch_size, image_size, image_size, 3))

trimap_f = tf.placeholder(tf.int32, shape=(train_batch_size, image_size, image_size, 1))
trimap_b = tf.placeholder(tf.int32, shape=(train_batch_size, image_size, image_size, 1))
trimap_i = tf.placeholder(tf.int32, shape=(train_batch_size, image_size, image_size, 1))

trimap_fbi_f = tf.cast(tf.divide(tf.concat([trimap_f, trimap_b, trimap_i], 3), tf.constant(127)), tf.float32)

distance_fi = tf.placeholder(tf.float32, shape=(train_batch_size,image_size,image_size,1))
distance_bi = tf.placeholder(tf.float32, shape=(train_batch_size,image_size,image_size,1))

distance_fi_f = tf.divide(distance_fi, 255.0)
distance_bi_f = tf.divide(distance_bi, 255.0)

cal_alpha  = tf.placeholder(tf.int32, shape = (train_batch_size,image_size,image_size,1))
diff_alpha = tf.placeholder(tf.int32, shape = (train_batch_size,image_size,image_size,1))
real_alpha  = tf.placeholder(tf.int32, shape = (train_batch_size,image_size,image_size,1))

training = tf.placeholder(tf.bool)
trainable = True

fbi_f = tf.divide(tf.cast(tf.concat([F, B, I], axis=3), tf.float32), 255.0)

cal_alpha_f = tf.cast(cal_alpha,tf.float32) / 255.0

input_concat = fbi_f

# conv1_1
with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 9, 64], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(input_concat, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv1_2
with tf.name_scope('conv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool1
pool1,arg1 = tf.nn.max_pool_with_argmax(conv1_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')
pool_parameters.append(arg1)

# conv2_1
with tf.name_scope('conv2_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv2_2
with tf.name_scope('conv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool2
pool2,arg2 = tf.nn.max_pool_with_argmax(conv2_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool2')
pool_parameters.append(arg2)

# conv3_1
with tf.name_scope('conv3_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv3_2
with tf.name_scope('conv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv3_3
with tf.name_scope('conv3_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_3 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool3
pool3,arg3 = tf.nn.max_pool_with_argmax(conv3_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool3')
pool_parameters.append(arg3)

# conv4_1
with tf.name_scope('conv4_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv4_2
with tf.name_scope('conv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv4_3
with tf.name_scope('conv4_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_3 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool4
pool4,arg4 = tf.nn.max_pool_with_argmax(conv4_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool4')
pool_parameters.append(arg4)

# conv5_1
with tf.name_scope('conv5_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                     trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv5_2
with tf.name_scope('conv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv5_3
with tf.name_scope('conv5_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_3 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool5
pool5,arg5 = tf.nn.max_pool_with_argmax(conv5_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool5')
pool_parameters.append(arg5)
# conv6_1
with tf.name_scope('conv6_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([2, 2, 512, 256], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv6_1 = tf.nn.relu(out, name='conv6_1')
    en_parameters += [kernel, biases]
#deconv6
with tf.variable_scope('deconv6') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 256, 512], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(conv6_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv6 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv6')

#deconv5_1/unpooling
deconv5_1 = unpool(deconv6,pool_parameters[-1])

#deconv5_2
with tf.variable_scope('deconv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 512], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(deconv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv5_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv5_2')

#deconv4_1/unpooling
deconv4_1 = unpool(deconv5_2,pool_parameters[-2])

#deconv4_2
with tf.variable_scope('deconv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 256], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(deconv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv4_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=True), name='deconv4_2')

#deconv3_1/unpooling
deconv3_1 = unpool(deconv4_2,pool_parameters[-3])

#deconv3_2
with tf.variable_scope('deconv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 256, 128], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(deconv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv3_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv3_2')

#deconv2_1/unpooling
deconv2_1 = unpool(deconv3_2,pool_parameters[-4])

#deconv2_2
with tf.variable_scope('deconv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 128, 64], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(deconv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv2_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv2_2')

#deconv1_1/unpooling
deconv1_1 = unpool(deconv2_2,pool_parameters[-5])

#deconv1_2
with tf.variable_scope('deconv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 64], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(deconv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv1_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv1_2')
#pred_alpha_matte
with tf.variable_scope('pred_alpha') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 1], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(deconv1_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    pred_mat_f = out
    pred_mat_f = tf.where(tf.greater(pred_mat_f,1),tf.ones_like(pred_mat_f),pred_mat_f)
    pred_mat_f = tf.where(tf.less(pred_mat_f,0),tf.zeros_like(pred_mat_f),pred_mat_f,name='res')



cal_alpha_x = pred_mat_f

diff_x = abs(cal_alpha_f - cal_alpha_x)
cou = tf.cast(tf.reduce_prod(tf.shape(cal_alpha_f)[:-1]), tf.float32)
cou = tf.Print(cou,[cal_alpha_f[0,0,0,:]],message="cou is ")
cou = tf.Print(cou,[cal_alpha_x[0,0,0,:]],message="cou is ")

mae = tf.divide(tf.reduce_sum(diff_x),cou)
mse = tf.divide(tf.reduce_sum(tf.pow(diff_x,2.0)),cou)
sad = tf.divide(tf.reduce_sum(diff_x),1000.0)

MAE_his = tf.summary.histogram('MAE_his',mae,family='stage1')
MSE_his = tf.summary.histogram('MSE_his',mse,family='stage1')
SAD_his = tf.summary.histogram('SAD_his',sad,family='stage1')


train_op   = tf.train.AdamOptimizer(learning_rate = 3e-3).minimize(mse)

saver = tf.train.Saver(max_to_keep=5)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
train_writer = tf.summary.FileWriter('/disk3/Graduate-design/train_log/train/')
with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    #initialize all parameters in vgg16
    idx = 0
    if pretrained_model:
        weights = np.load('/disk3/Graduate-design/model/vgg16_weights.npz')
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i,k
            if i == 28:
                break
            if k == 'conv1_1_W':  
                para1 = np.concatenate([weights[k] for _ in range(3)], axis=2)
                sess.run(en_parameters[i].assign(para1))
            else:
                if k=='fc6_W' or k == 'fc6_b':
                    continue
                    tmp = np.reshape(weights[k],(7,7,512,4096))
                    sess.run(en_parameters[i].assign(tmp))
                else:
                    sess.run(en_parameters[i].assign(weights[k]))
        print('finish loading vgg16 model')
    else:
        print('Restoring pretrained model...')
        saver.restore(sess,tf.train.latest_checkpoint('/disk3/Graduate-design/model/',latest_filename='latestcheckpoint_file'))
        if is_train:
            idx = np.load('/disk3/Graduate-design/model/idx.npy') + 1

    count_ = 0
    for _ in range(10000):
        if is_train:

            batch_F = np.array([cv2.imread("/disk3/Graduate-design/data/method2/F/{:0>6}.png".format(idx*10 + i)) for i in range(10)])
            batch_B = np.array([cv2.imread("/disk3/Graduate-design/data/method2/B/{:0>6}.png".format(idx*10 + i)) for i in range(10)])
            batch_I = np.array([cv2.imread("/disk3/Graduate-design/data/method2/I/{:0>6}.png".format(idx*10 + i)) for i in range(10)])

            batch_trimap_f = np.array([cv2.imread("/disk3/Graduate-design/data/method2/trimap_f/{:0>6}.png".format(idx*10 + i)) for i in range(10)])[:,:,:,0:1]
            batch_trimap_b = np.array([cv2.imread("/disk3/Graduate-design/data/method2/trimap_b/{:0>6}.png".format(idx*10 + i)) for i in range(10)])[:,:,:,0:1]
            batch_trimap_i = np.array([cv2.imread("/disk3/Graduate-design/data/method2/trimap_i/{:0>6}.png".format(idx*10 + i)) for i in range(10)])[:,:,:,0:1]

            batch_distance_fi = np.array([np.load("/disk3/Graduate-design/data/method2/distance_fi/{:0>6}.npy".format(idx*10 + i)) for i in range(10)])
            batch_distance_bi = np.array([np.load("/disk3/Graduate-design/data/method2/distance_bi/{:0>6}.npy".format(idx*10 + i)) for i in range(10)])

            batch_cal_alpha  = np.array([cv2.imread("/disk3/Graduate-design/data/method2/cal_alpha/{:0>6}.png".format(idx*10 + i)) for i in range(10)])[:,:,:,0:1]
            batch_real_alpha = np.array([cv2.imread("/disk3/Graduate-design/data/method2/real_alpha/{:0>6}.png".format(idx*10 + i)) for i in range(10)])[:,:,:,0:1]

            batch_diff_alpha = np.array([np.load("/disk3/Graduate-design/data/method2/diff_alpha/{:0>6}.npy".format(idx*10 + i)) for i in range(10)])[:,:,:,0:1]

            # construct feed_dict for stage 1 training
            feed = {F:batch_F, B:batch_F, I:batch_I, training:True,
                    trimap_f:batch_trimap_f, trimap_b:batch_trimap_b, trimap_i:batch_trimap_i,
                    distance_fi:batch_distance_fi, distance_bi:batch_distance_bi,
                    cal_alpha:batch_cal_alpha, real_alpha:batch_real_alpha, diff_alpha:batch_diff_alpha}
            pred_alpha = tf.get_default_graph().get_tensor_by_name('pred_alpha/res:0')


            _,mae_,sad_,mse_,diff_ = sess.run([train_op,mae,sad,mse,diff_x],feed_dict = feed)
            print("the step is %06d  mae is %f  sad is %f  mse is %f" % (idx,mae_,sad_,mse_))



        if idx+1 % 100 == 0:
            saver.save(sess = sess, save_path = '/disk3/Graduate-design/model/model',
                global_step = idx, latest_filename='latestcheckpoint_file')
            np.save('/disk3/Graduate-design/model/idx.npy',np.array(idx))
        idx += 1


