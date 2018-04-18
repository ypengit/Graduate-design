import tensorflow as tf
import numpy as np
import cv2
import glob

file_pathes = ["/tmp/deep_matting/input_training_lowres/*.png",
        "/tmp/deep_matting/trimap_training_lowres/Trimap1/*.png",
        "/tmp/deep_matting/gt_training_lowres/*.png"
        ]
pic_num = 26
ts = np.ndarray(shape=(3,pic_num), dtype=tf.Tensor)
for t,i in enumerate(file_pathes):
    files = glob.glob(i)
    files.sort()
    for ix,j in enumerate(files):
        ts[t][ix] = tf.constant(cv2.imread(j),dtype=tf.uint8)


#def block(idx, t):
#    with tf.name_scope(t):
#        if(t.equal("I"))
#        ts[0,idx]


def next(batch):
    idx = tf.random_uniform([batch], 0, pic_num, dtype=tf.int32)
    with tf.Session() as sess:
        # print sess.run(idx)
        print sess.run(idx)
        #print ts
        #print sess.run(a).shape
        #print sess.run(tf.gather(a,[idx])).shape

if __name__=="__main__":
    next(10)
