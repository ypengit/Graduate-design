import tensorflow as tf
import os
import cv2
import numpy as np
import pdb

os.environ['CUDA_VISIABLE_DEVICE'] = '0'
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('/disk3/Graduate-design/model/model-19900.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/disk3/Graduate-design/model/','latestcheckpoint_file'))
    graph = tf.get_default_graph()
    rgb        = graph.get_tensor_by_name('Placeholder:0')
    trimap     = graph.get_tensor_by_name('Placeholder_2:0')
    pred_mat_s = graph.get_tensor_by_name('Placeholder_3:0')
    trainning  = graph.get_tensor_by_name('Placeholder_5:0')
    pred_mat = graph.get_tensor_by_name('pred_alpha/res:0')
    pred_mat_f = graph.get_tensor_by_name('refinement/ref4/res:0')
    rg = cv2.imread("/tmp/deep_matting/input_training_lowres/GT04.png")
    tr = cv2.imread("/tmp/deep_matting/trimap_training_lowres/Trimap1/GT04.png")
    sh = rg.shape
    rg_c = np.pad(rg,[[0,320],[0,320],[0,0]],'constant',constant_values=0)
    tr_c = np.pad(tr,[[0,320],[0,320],[0,0]],'constant',constant_values=0)
    print rg_c.shape,tr_c.shape
    al_f = np.zeros_like(tr_c[:,:,0:1])
    idx = 0
    for y in range(0,sh[0],320):
        for x in range(0,sh[1],320):
            rg_ = np.stack([rg_c[y:y+320,x:x+320,:] for _ in range(10)])
            tr_ = np.stack([tr_c[y:y+320,x:x+320,0:1] for _ in range(10)])
            print 'idx is %02d' % (idx)
            feed = {rgb:rg_,trimap:tr_,trainning:True}
            pred_sg1 = sess.run([pred_mat],feed_dict=feed)[0]
            feed_f = {rgb:rg_,trimap:tr_,pred_mat_f:pred_sg1,trainning:True}
            pred_sg2 = sess.run([pred_mat_f],feed_dict=feed_f)
            idx += 1
            # pdb.set_trace()
            al_f[y:y+320,x:x+320,:] = pred_sg2[0][0]
    al_f = al_f[0:sh[0],0:sh[1],:]
    cv2.imwrite('/home/yubin/Graduate-design/Matting_with_deep_learning/res.png',al_f*255)
