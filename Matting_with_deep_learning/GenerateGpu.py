import tensorflow as tf
import numpy as np
import cv2
import glob



file_pathes = ["/tmp/deep_matting/input_training_lowres/*.png",
        "/tmp/deep_matting/trimap_training_lowres/Trimap1/*.png",
        "/tmp/deep_matting/gt_training_lowres/*.png"
        ]
pic_num = 26
width = 20
height = 20
# data
dt = np.ndarray(shape=(3,pic_num),dtype=np.ndarray)
# tensors
ts = np.ndarray(shape=(3,pic_num),dtype=tf.Tensor)
# read files from directory
for t,i in enumerate(file_pathes):
    files = glob.glob(i)
    files.sort()
    for ix,j in enumerate(files):
        img = cv2.imread(j)
        dt[t][ix],ts[t][ix] = img,tf.constant(img)
# record the shape of training data
shape = tf.constant(np.array([x.shape for x in dt[0]],dtype=np.int32),tf.int32)
#for x in ts[:]:
#    print x

def block(idx, t):
    with tf.name_scope(t):
        # Get the shape of batch according to idx
        pos_pre = tf.to_float(tf.gather(shape,idx))
        # Get random pos with pos_pre
        pos,_ = tf.split(tf.to_int32(tf.round(tf.multiply(pos_pre,tf.random_uniform(pos_pre.shape,minval=0.0,maxval=1.0))),name="pos"),[2,1],1)
        # concat the idx and pos
        idx_pos = tf.concat([tf.expand_dims(idx,1),pos],1)
        # print ts[0][0]
        # print tf.slice(ts[0][0],[0,0,0],[1,1,1])
        print [tf.slice(tf.pad(ts[0][x[0]],[[height/2,height/2],[width/2,width/2],[0,0]],constant_values=-255),[x[1]-width/2,x[2]-width/2,0],[20,20,3]) for x in idx_pos.eval()]

        return idx_pos



def next(batch):
    idx = np.random.randint(0,pic_num,[batch],dtype=np.int32)
    idx = tf.to_int32(idx,name="idx")
    with tf.Session() as sess:
        # print sess.run(idx)
        # print sess.run(idx)
        # print sess.run(tf.concat([pos,idx],0))
        # print pos
        print sess.run(block(idx,"F"))
        print sess.run(block(idx,"B"))
        print sess.run(block(idx,"I"))
        print sess.run(shape)
        #print sess.run(shape)
        #print ts
        #print sess.run(a).shape
        #print sess.run(tf.gather(a,[idx])).shape

if __name__=="__main__":
    next(10)
