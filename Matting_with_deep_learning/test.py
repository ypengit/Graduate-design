import tensorflow as tf
import Generate
saver_path = "./model_save/"
saver_file = saver_path + "model" + ".meta"
batch_size = 128
saver = tf.train.import_meta_graph(saver_file) 
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(saver_path))
    batch = Generate.next(batch_size)
    graph = tf.get_default_graph()
    v1 = graph.get_tensor_by_name('Outer/fc13/x:0')
    print(sess.run(v1, feed_dict=batch))
