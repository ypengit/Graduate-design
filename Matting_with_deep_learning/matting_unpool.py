import tensorflow as tf
import cv2
import numpy as np
from matting import load_path,load_data,load_alphamatting_data,load_validation_data,unpool
import os
import pdb
from scipy import misc
os.environ['CUDA_VISIBLE_DEVICES']='1'

image_size = 320
train_batch_size = 10
max_epochs = 1000000
hard_mode = False

#checkpoint file path
pretrained_model = False
#pretrained_model = False
test_dir = './alhpamatting'
test_outdir = './test_predict'
#validation_dir = '/data/gezheng/data-matting/new2/validation'

#pretrained_vgg_model_path
model_path = './vgg16_weights.npz'
log_dir = 'matting_log'

dataset_alpha = 'train_data/alpha'
dataset_eps = 'train_data/eps'




image_batch = tf.placeholder(tf.float32, shape=(train_batch_size,image_size,image_size,3))
GT_matte_batch = tf.placeholder(tf.float32, shape = (train_batch_size,image_size,image_size,1))
GT_trimap = tf.placeholder(tf.float32, shape = (train_batch_size,image_size,image_size,1))
training = tf.placeholder(tf.bool)

en_parameters = []
pool_parameters = []

b_RGB = tf.identity(image_batch,name = 'b_RGB')
b_trimap = tf.identity(GT_trimap,name = 'b_trimap')
b_GTmatte = tf.identity(GT_matte_batch,name = 'b_GTmatte')

b_input = tf.concat([b_RGB,b_trimap],3)

# conv1_1
with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 4, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(b_input, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv1_2
with tf.name_scope('conv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool1
pool1,arg1 = tf.nn.max_pool_with_argmax(conv1_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')
pool_parameters.append(arg1)

# conv2_1
with tf.name_scope('conv2_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv2_2
with tf.name_scope('conv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool2
pool2,arg2 = tf.nn.max_pool_with_argmax(conv2_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool2')
pool_parameters.append(arg2)

# conv3_1
with tf.name_scope('conv3_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv3_2
with tf.name_scope('conv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv3_3
with tf.name_scope('conv3_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_3 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool3
pool3,arg3 = tf.nn.max_pool_with_argmax(conv3_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool3')
pool_parameters.append(arg3)

# conv4_1
with tf.name_scope('conv4_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv4_2
with tf.name_scope('conv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv4_3
with tf.name_scope('conv4_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_3 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool4
pool4,arg4 = tf.nn.max_pool_with_argmax(conv4_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool4')
pool_parameters.append(arg4)

# conv5_1
with tf.name_scope('conv5_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                         stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                     trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv5_2
with tf.name_scope('conv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv5_3
with tf.name_scope('conv5_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_3 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool5
pool5,arg5 = tf.nn.max_pool_with_argmax(conv5_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool5')
pool_parameters.append(arg5)
# conv6_1
with tf.name_scope('conv6_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([7, 7, 512, 4096], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv6_1 = tf.nn.relu(out, name='conv6_1')
    en_parameters += [kernel, biases]
#deconv6
with tf.variable_scope('deconv6') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv6_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv6 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv6')

#deconv5_1/unpooling
deconv5_1 = unpool(deconv6,pool_parameters[-1])

#deconv5_2
with tf.variable_scope('deconv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv5_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv5_2')

#deconv4_1/unpooling
deconv4_1 = unpool(deconv5_2,pool_parameters[-2])

#deconv4_2
with tf.variable_scope('deconv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv4_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=True), name='deconv4_2')

#deconv3_1/unpooling
deconv3_1 = unpool(deconv4_2,pool_parameters[-3])

#deconv3_2
with tf.variable_scope('deconv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 256, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv3_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv3_2')

#deconv2_1/unpooling
deconv2_1 = unpool(deconv3_2,pool_parameters[-4])

#deconv2_2
with tf.variable_scope('deconv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 128, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv2_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv2_2')

#deconv1_1/unpooling
deconv1_1 = unpool(deconv2_2,pool_parameters[-5])

#deconv1_2
with tf.variable_scope('deconv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv1_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv1_2')
#pred_alpha_matte
with tf.variable_scope('pred_alpha') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 1], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv1_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    pred_mattes = out
    pred_mattes = tf.where(tf.equal(tf.cast(b_trimap,tf.int32),255),tf.ones_like(pred_mattes),pred_mattes)
    pred_mattes = tf.where(tf.equal(tf.cast(b_trimap,tf.int32),0),tf.zeros_like(pred_mattes),pred_mattes)
    pred_mattes = tf.where(tf.greater(pred_mattes,1),tf.ones_like(pred_mattes),pred_mattes)
    pred_mattes = tf.where(tf.less(pred_mattes,0),tf.zeros_like(pred_mattes),pred_mattes,name='res')


pred_mattes_out = tf.reshape(pred_mattes,tf.shape(GT_matte_batch))


coun = tf.reduce_sum(tf.where(tf.equal(tf.cast(b_trimap,tf.int32),128*tf.ones_like(b_trimap,tf.int32)),tf.ones_like(b_trimap),tf.zeros_like(b_trimap)))

total_loss = tf.reduce_sum(tf.abs(tf.subtract(pred_mattes_out,GT_matte_batch/255.0)))/coun
sad = tf.reduce_sum(tf.abs(tf.subtract(pred_mattes_out,GT_matte_batch/255.0)))/1000.0
mse = tf.reduce_sum(tf.pow(tf.abs(tf.subtract(pred_mattes_out,GT_matte_batch/255.0)),2.0))/coun

global_step = tf.Variable(0,trainable=False)


train_op = tf.train.AdamOptimizer(learning_rate = 1e-5).minimize(mse,global_step = global_step)

saver = tf.train.Saver(tf.trainable_variables() , max_to_keep = 1)

coord = tf.train.Coordinator()
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(coord=coord,sess=sess)
    batch_num = 0
    epoch_num = 0
    #initialize all parameters in vgg16
    if not pretrained_model:
        weights = np.load(model_path)
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
        saver.restore(sess,tf.train.latest_checkpoint('./model'))
    sess.graph.finalize()

    for idx in range(10000):
        # pdb.set_trace()
        batch_RGBs   = np.array([cv2.imread("/disk3/Graduate-design/data/rgb/{:0>6}.png".format(idx*10 + i)) for i in range(10)])
        batch_alphas = np.array([cv2.imread("/disk3/Graduate-design/data/alpha/{:0>6}.png".format(idx*10 + i)) for i in range(10)])[:,:,:,0:1]
        batch_trimaps= np.array([cv2.imread("/disk3/Graduate-design/data/trimap/{:0>6}.png".format(idx*10 + i)) for i in range(10)])[:,:,:,0:1]
        feed = {image_batch:batch_RGBs, GT_matte_batch:batch_alphas,GT_trimap:batch_trimaps,training:True}
        _,loss,sad_,mse_ = sess.run([train_op,total_loss,sad,mse],feed_dict = feed)
        print('step is %04d loss is %f, sad is %f, mse is %f' %(idx,loss,sad_,mse_))
        if idx % 10 == 0:
            ix = idx / 10
            rgb=np.array([cv2.imread("/disk3/Graduate-design/test/rgb/{:0>6}.png".format(ix*10 + i)) for i in range(10)])
            alpha=np.array([cv2.imread("/disk3/Graduate-design/test/alpha/{:0>6}.png".format(ix*10 + i)) for i in range(10)])[:,:,:,0:1]
            tr=np.array([cv2.imread("/disk3/Graduate-design/test/trimap/{:0>6}.png".format(ix*10 + i)) for i in range(10)])[:,:,:,0:1]
            for ixx,pic in enumerate(sess.run(tf.get_default_graph().get_tensor_by_name("pred_alpha/res:0"),feed_dict={image_batch:rgb,GT_matte_batch:alpha,GT_trimap:tr,training:True})*255):
                cv2.imwrite("./res/{:0>4}.png".format(ix*10 + ixx),pic)
            print sess.run([total_loss,sad,mse],feed_dict={image_batch:rgb,GT_matte_batch:alpha,GT_trimap:tr,training:True})
        batch_num += 1
    batch_num = 0
    epoch_num += 1


