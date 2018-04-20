import tensorflow as tf
import numpy as np
import cv2
import glob
file_pathes = ["/tmp/deep_matting/input_training_lowres/*.png",
        "/tmp/deep_matting/trimap_training_lowres/Trimap1/*.png",
        "/tmp/deep_matting/gt_training_lowres/*.png"
        ]
pic_num = 26
width = 30
height = 30
# data
dt = np.ndarray(shape=(3,pic_num),dtype=np.ndarray)
# tensors
ts = np.ndarray(shape=(3,pic_num),dtype=tf.Tensor)
# read files from directory
for t,i in enumerate(file_pathes):
    files = glob.glob(i)
    files.sort()
    for ix,j in enumerate(files):
        img = cv2.imread(j)/255.0
        dt[t][ix],ts[t][ix] = img,tf.constant(img)
# record the shape of training data
shape = np.array([x.shape for x in dt[0]],dtype=np.int32)
# pad the source Tensor
pad = np.array([tf.pad(x,[[height/2,height/2],[width/2,width/2],[0,0]],constant_values=-255) for x in ts[0]])
def block(idx, t):
    with tf.name_scope(t):
        # Get the shape of batch according to idx
        # Get random pos with pos_pre
        pre_pos = shape[idx][:,:-1]
        pos = np.floor(np.multiply(pre_pos,np.random.random(pre_pos.shape))).astype(np.int32)
        # concat the idx and pos
        idx_pos = tf.concat([tf.expand_dims(idx,1),pos],1,name="idx_pos")
        res = tf.to_float(tf.stack([tf.slice(pad[x[0]], [x[1],x[2],0],[height+1,width+1,3]) for x in idx_pos.eval()]))
        val = tf.stack([ts[0][x[0]][x[1]][x[2]] for x in idx_pos.eval()],name="val")
        # stack a batch of data
        return res
def real_alpha():
    idx_pos = tf.get_default_graph().get_tensor_by_name("I/idx_pos:0")
    return tf.to_float(np.array([dt[2][x[0]][x[1]][x[2]][0] for x in idx_pos.eval()]))
def cal_alpha():
    F = tf.get_default_graph().get_tensor_by_name("F/val:0")
    B = tf.get_default_graph().get_tensor_by_name("B/val:0")
    I = tf.get_default_graph().get_tensor_by_name("I/val:0")
    return tf.divide(tf.reduce_sum(tf.to_float(tf.multiply(tf.subtract(I,B),tf.subtract(F,B))),1),tf.reduce_sum(tf.add(tf.to_float(tf.pow(tf.subtract(F,B),2.0)),0.0001),1))
def next(batch):
    idx = np.random.randint(0,pic_num,[batch],dtype=np.int32)
    F = block(idx,"F")
    B = block(idx,"B")
    I = block(idx,"I")
    idx_pos = tf.get_default_graph().get_tensor_by_name("I/idx_pos:0")
    target = tf.expand_dims((cal_alpha() - real_alpha()),1)
    return F,B,I,target
if __name__ == "__main__":
    with tf.Session() as sess:
        for i in range(10):
            F,B,I,target = next(100)
            print sess.run(F)
            print sess.run(B)
            print sess.run(I)
            print sess.run(target)
