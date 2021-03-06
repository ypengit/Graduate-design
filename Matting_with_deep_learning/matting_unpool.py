import tensorflow as tf
import cv2
import numpy as np
import os
import pdb
from scipy import misc
os.environ['CUDA_VISIBLE_DEVICES']='0'

image_size = 320
train_batch_size = 10

# choose weather to load data to sess
pretrained_model = False
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

def dropout(keep_prob, is_training, _input):
    output = tf.cond(is_training,
            lambda: tf.nn.dropout(_input, keep_prob),
            lambda: _input
            )
    return output


rgb    = tf.placeholder(tf.int32, shape = (train_batch_size,image_size,image_size,3))
alpha  = tf.placeholder(tf.int32, shape = (train_batch_size,image_size,image_size,1))
trimap = tf.placeholder(tf.int32, shape = (train_batch_size,image_size,image_size,1))
pred_mat_s = tf.placeholder(tf.float32, shape = (train_batch_size, image_size, image_size,1))
rgb_sm = tf.summary.image('rgb',tf.cast(rgb,tf.float32),max_outputs=5)
alpha_sm = tf.summary.image('alpha',tf.cast(alpha,tf.float32),max_outputs=5)
trimap_sm = tf.summary.image('trimap',tf.cast(trimap,tf.float32),max_outputs=5)

pred_mat_s_sm = tf.summary.image('pred_mat_s',pred_mat_s,max_outputs=5)
keep_prob = tf.placeholder(tf.float32)

training = tf.placeholder(tf.bool)
trainable= True

input_concat = tf.divide(tf.cast(tf.concat([rgb,trimap],3),tf.float32),255.0)

# conv1_1
with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 4, 64], dtype=tf.float32,
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
    kernel = tf.Variable(tf.truncated_normal([7, 7, 512, 4096], dtype=tf.float32,
                                 stddev=1e-1), name='weights',trainable=trainable)
    conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                         trainable=trainable, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv6_1 = tf.nn.relu(out, name='conv6_1')
    en_parameters += [kernel, biases]
#deconv6
with tf.variable_scope('deconv6') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, 512], dtype=tf.float32,
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
    pred_mat = out
    pred_mat = tf.where(tf.equal(trimap,255),tf.ones_like(pred_mat),pred_mat)
    pred_mat = tf.where(tf.equal(trimap,  0),tf.zeros_like(pred_mat),pred_mat)
    pred_mat = tf.where(tf.greater(pred_mat,1),tf.ones_like(pred_mat),pred_mat)
    pred_mat = tf.where(tf.less(pred_mat,0),tf.zeros_like(pred_mat),pred_mat,name='res')
    pred_mat_sm = tf.summary.image('pred_mat',pred_mat,max_outputs=5)


with tf.name_scope('refinement') as scope:
    with tf.variable_scope('ref1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3,  5,64], dtype=tf.float32,
                                     stddev=1e-1), name='weights',trainable=trainable)
        conv = tf.nn.conv2d(tf.concat([pred_mat_s,tf.divide(tf.cast(rgb,tf.float32),255.0),
            tf.divide(tf.cast(trimap,tf.float32),255.0)],axis=3), kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        ref1_ = tf.nn.relu(tf.nn.bias_add(conv, biases))
        ref1 = dropout(keep_prob, training, ref1_)
    with tf.variable_scope('ref2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64,64], dtype=tf.float32,
                                     stddev=1e-1), name='weights',trainable=trainable)
        conv = tf.nn.conv2d(ref1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        ref2_ = tf.nn.relu(tf.nn.bias_add(conv, biases) + ref1)
        ref2 = dropout(keep_prob, training, ref2_)

    with tf.variable_scope('ref3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64,64], dtype=tf.float32,
                                     stddev=1e-1), name='weights',trainable=trainable)
        conv = tf.nn.conv2d(ref2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        ref3_ = tf.nn.relu(tf.nn.bias_add(conv, biases) + ref1)
        ref3 = dropout(keep_prob, training, ref3_)
    with tf.variable_scope('ref3') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 64,64], dtype=tf.float32,
                                     stddev=1e-1), name='weights',trainable=trainable)
        conv = tf.nn.conv2d(ref3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        ref4_ = tf.nn.relu(tf.nn.bias_add(conv, biases) + ref1)
        ref4 = dropout(keep_prob, training, ref4_)
    with tf.variable_scope('ref4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 1], dtype=tf.float32,
                                     stddev=1e-1), name='weights',trainable=trainable)
        conv = tf.nn.conv2d(ref4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        ref5 = tf.nn.bias_add(conv, biases)
        pred_mat_f = ref5
        pred_mat_f = tf.where(tf.equal(trimap,255),tf.ones_like(pred_mat_f),pred_mat_f)
        pred_mat_f = tf.where(tf.equal(trimap,  0),tf.zeros_like(pred_mat_f),pred_mat_f)
        pred_mat_f = tf.where(tf.greater(pred_mat_f,1),tf.ones_like(pred_mat_f),pred_mat_f)
        pred_mat_f = tf.where(tf.less(pred_mat_f,0),tf.zeros_like(pred_mat_f),pred_mat_f,name='res')

pred_mat_f_sm = tf.summary.image('pred_mat_f',pred_mat_f,max_outputs=5)
pred_mat_f_l = tf.where(tf.equal(trimap,255),tf.ones_like(pred_mat_f),pred_mat_f)
pred_mat_f_l = tf.where(tf.equal(trimap,  0),tf.zeros_like(pred_mat_f_l),pred_mat_f_l,name='res')
pred_mat_f_l_sm = tf.summary.image('pred_mat_f_l',tf.multiply(pred_mat_f_l,255.0),max_outputs=5)



alpha_f = tf.divide(tf.cast(alpha,tf.float32),255.0)


cou = tf.cast(tf.reduce_sum(tf.where(tf.equal(trimap,128),tf.ones_like(trimap),tf.zeros_like(trimap))), tf.float32)

diff   = tf.abs(tf.subtract(pred_mat,alpha_f))
diff_f = tf.abs(tf.subtract(pred_mat_f,alpha_f))

mae = tf.divide(tf.reduce_sum(diff),cou)
mse = tf.divide(tf.reduce_sum(tf.pow(diff,2.0)),cou)
sad = tf.divide(tf.reduce_sum(diff),1000.0)

MAE_his = tf.summary.histogram('MAE_his',mae,family='stage1')
MSE_his = tf.summary.histogram('MSE_his',mse,family='stage1')
SAD_his = tf.summary.histogram('SAD_his',sad,family='stage1')


MAE_sm = tf.summary.scalar('MAE',mae,family='stage1')
MSE_sm = tf.summary.scalar('MSE',mse,family='stage1')
SAD_sm = tf.summary.scalar('SAD',sad,family='stage1')

mae_f = tf.divide(tf.reduce_sum(diff_f),cou)
mse_f = tf.divide(tf.reduce_sum(tf.pow(diff_f,2.0)),cou)
sad_f = tf.divide(tf.reduce_sum(diff_f),1000.0)

MAE_f_sm = tf.summary.scalar('MAE_f',mae_f,family='stage2')
MSE_f_sm = tf.summary.scalar('MSE_f',mse_f,family='stage2')
SAD_f_sm = tf.summary.scalar('SAD_f',sad_f,family='stage2')

train_op   = tf.train.AdamOptimizer(learning_rate = 1e-5).minimize(mse)
train_op_f = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(mse_f)

pre = tf.contrib.slim.get_variables_to_restore()
saver = tf.train.Saver(pre, max_to_keep=5)
saver1 = tf.train.Saver(max_to_keep=5)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
merged_st1 = tf.summary.merge([rgb_sm,alpha_sm,trimap_sm,MAE_sm,MSE_sm,SAD_sm,pred_mat_sm,MAE_his,MSE_his,SAD_his])
merged_st2 = tf.summary.merge([pred_mat_s_sm,pred_mat_f_sm,MAE_f_sm,MSE_f_sm,SAD_f_sm,pred_mat_f_l_sm])
train_writer = tf.summary.FileWriter('/disk3/Graduate-design/train_log/train/')
test_writer = tf.summary.FileWriter('/disk3/Graduate-design/train_log/test/')
with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    #initialize all parameters in vgg16
    idx = 0
    if pretrained_model:
        weights = np.load('/disk3/Graduate-design/model/vgg16_weights.npz')
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i == 28:
                break
            if k == 'conv1_1_W':  
                sess.run(en_parameters[i].assign(np.concatenate([weights[k],np.zeros([3,3,1,64])],axis = 2)))
            else:
                if k=='fc6_W':
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
    s_ = 0
    c_ = 0
    sum_ = 0
    c_2 = 0
    sum_2 = 0
    for _ in range(10000):
        if is_train:
            batch_rgb   = np.array([cv2.imread("/disk3/Graduate-design/data/rgb/{:0>6}.png".format(idx*10 + i)) for i in range(10)])
            batch_alpha = np.array([cv2.imread("/disk3/Graduate-design/data/alpha/{:0>6}.png".format(idx*10 + i)) for i in range(10)])[:,:,:,0:1]
            batch_trimap= np.array([cv2.imread("/disk3/Graduate-design/data/trimap/{:0>6}.png".format(idx*10 + i)) for i in range(10)])[:,:,:,0:1]

            # construct feed_dict for stage 1 training
            feed = {rgb:batch_rgb, alpha:batch_alpha,trimap:batch_trimap,training:True}
            pred_alpha = tf.get_default_graph().get_tensor_by_name('pred_alpha/res:0')
            _,summary,mae_,sad_,mse_,pred_mat_s1 = sess.run([train_op,merged_st1,mae,sad,mse,pred_alpha],feed_dict = feed)
            print('step is %06d MAE is %f, SAD is %f, MSE is %f' %(idx,mae_,sad_,mse_))
            if idx % 5 == 0:
                train_writer.add_summary(summary,idx)

            # if the idx is greater than 1000, start to train stage2, or it will
            # make no sense
            if idx > 2000:
                # construct feed_dict for stage 2 training
                pred_f = tf.get_default_graph().get_tensor_by_name('refinement/ref4/res:0')
                feed_f = {rgb:batch_rgb,alpha:batch_alpha,trimap:batch_trimap,pred_mat_s:pred_mat_s1,keep_prob:0.5,training:True}
                _,summary,mae_fs,sad_fs,mse_fs,pred_mat_s2 = sess.run([train_op_f,merged_st2,mae_f,sad_f,mse_f,pred_f],feed_dict = feed_f)
                for ix in range(10):
                    cv2.imwrite("res{:06}.png".format(idx*10+ix),255*pred_mat_s2[ix])
                    cv2.imwrite("alpha{:06}.png".format(idx*10+ix),batch_alpha[ix])
                    cv2.imwrite("trimap{:06}.png".format(idx*10+ix),batch_trimap[ix])

                # show the loss of stage 1 and stage 2
                count_ += 1
                s_ += mse_fs
                print('step is %06d MAE is %f, SAD is %f, MSE is %f, AVERAGE mse is %f' %(idx,mae_fs,sad_fs,mse_fs,s_/(count_ + 0.01)))
                print sess.run([cou, tf.shape(diff_f), diff_f],feed_dict=feed_f)
                pdb.set_trace()
            if idx % 5 == 0:
                train_writer.add_summary(summary,idx)

        if idx % 10 == 0:
            ix = idx / 10
            rg_t = np.array([cv2.imread("/disk3/Graduate-design/test/rgb/{:0>6}.png".format(ix*10 + i)) for i in range(10)])
            al_t = np.array([cv2.imread("/disk3/Graduate-design/test/alpha/{:0>6}.png".format(ix*10 + i)) for i in range(10)])[:,:,:,0:1]
            tr_t = np.array([cv2.imread("/disk3/Graduate-design/test/trimap/{:0>6}.png".format(ix*10 + i)) for i in range(10)])[:,:,:,0:1]
            feed_t = {rgb:rg_t,alpha:al_t,trimap:tr_t,training:False, keep_prob:0.5}
            pre_t = tf.get_default_graph().get_tensor_by_name("pred_alpha/res:0")
            st1 = sess.run(pre_t,feed_dict=feed_t)
            pred_f = tf.get_default_graph().get_tensor_by_name('refinement/ref4/res:0')
            feed_t.update({pred_mat_s:st1})
            for ixx,pic in enumerate(sess.run(pred_f,feed_dict=feed_t)):
                cv2.imwrite("./res/{:0>6}.png".format(ix*10 + ixx),pic*255)
            sum_ += sess.run(mse, feed_dict=feed_t)
            c_ += 1
            print 'test st1 step is %06d '%(idx),sess.run([mae,mse,sad],feed_dict=feed_t), " test average is %f" % (sum_ / (c_ + 0.01))
            summary = sess.run(merged_st1, feed_dict=feed_t)
            if idx % 5 == 0:
                test_writer.add_summary(summary,idx)
            feed_t.update({pred_mat_s:st1})
            sum_2 += sess.run(mse_f, feed_dict=feed_t)
            c_2 += 1
            print 'test st2 step is %06d '%(idx),sess.run([mae_f, mse_f, sad_f],feed_dict=feed_t), '  test average is %f' % (sum_2 / (c_2 + 0.01))
            summary = sess.run(merged_st2, feed_dict=feed_t)
            if idx % 5 == 0:
                test_writer.add_summary(summary,idx)
        if idx % 500 == 0:
            saver1.save(sess = sess, save_path = '/disk3/Graduate-design/model/model',
                global_step = idx, latest_filename='latestcheckpoint_file')
            np.save('/disk3/Graduate-design/model/idx.npy',np.array(idx))
        idx += 1


