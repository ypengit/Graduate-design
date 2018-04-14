import tensorflow as tf
import cv2
import numpy as np
import Generate

print Generate.data['train'][-1].shape
width, height, _ = Generate.data['train'][-1].shape

res = np.zeros([width, height, 3])

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)

saver_file = "/tmp/deep_matting/model_save/model"

with tf.Session(config=config, graph=tf.get_default_graph()) as sess:
    saver = tf.train.import_meta_graph(saver_file + ".meta")
    saver.restore(sess, saver_file)
    for x in range(width):
        for y in range(height):
            if(np.equal(Generate.data['trimap'][-1][x][y],
                np.array([128, 128, 128])).all()):
                # Get the block of I
                data_pics = {}
                data_pics['pos_I'] = {"x":x, "y":y}
                data_pics['I'] = Generate.get_block(-1, data_pics['pos_I'])
                I_test = [data_pics['I']]
                # Get the block of F and B, then do the prediction
                cal_alpha_diff_pre = 1.0
                pre_pos = {}
                pre_pos['pos_F'] = {'x':0, 'y':0}
                pre_pos['pos_B'] = {'x':0, 'y':0}
                dividual_alpha_diff_pre = 1.0
                loss = cal_alpha_diff_pre + dividual_alpha_diff_pre
                for i in range(1000):
                    # generate random pos according the shape of pictures
                    data_pics['pos_F'] = Generate.rand_pos(Generate.data['train'][-1].shape, 'F', -1)
                    data_pics['pos_B'] = Generate.rand_pos(Generate.data['train'][-1].shape, 'B', -1)
                    # get the block with pos
                    data_pics['F'] = Generate.get_block(-1, data_pics['pos_F'])
                    data_pics['B'] = Generate.get_block(-1, data_pics['pos_B'])
                    I_test = np.array([data_pics['I']])
                    F_test = np.array([data_pics['F']])
                    B_test = np.array([data_pics['B']])
                    I_val = Generate.get_val(-1, data_pics['pos_I'])
                    F_val = Generate.get_val(-1, data_pics['pos_F'])
                    B_val = Generate.get_val(-1, data_pics['pos_B'])
                    cal_dividual_alpha_diff = sum((I_val - B_val)/(F_val - B_val + 0.001)/255.0)/3
                    [[cal_alpha_diff]] = sess.run(tf.get_default_graph().get_tensor_by_name("fc13/x:0"), feed_dict={"Placeholder:0":F_test, "Placeholder_1:0":B_test, "Placeholder_2:0":I_test})
                    cal_loss = cal_dividual_alpha_diff + cal_alpha_diff
                    cal_alpha = Generate.cal_alpha(F_val, B_val, I_val)
                    # print i,cal_alpha,cal_dividual_alpha_diff,cal_alpha_diff
                    if(cal_loss < loss):
                        loss = cal_alpha_diff_pre
                        res[x][y][0] = cal_alpha
                        res[x][y][1] = cal_alpha
                        res[x][y][2] = cal_alpha
                print loss
            if(np.equal(Generate.data['trimap'][-1][x][y],
                np.array([255, 255, 255])).all()):
                res[x][y] = np.array([1.0, 1.0, 1.0])
            if(np.equal(Generate.data['trimap'][-1][x][y],
                np.array([0  , 0  , 0  ])).all()):
                res[x][y] = np.array([0.0, 0.0, 0.0])
            print("width=%04d,height=%04d,alpha=%.6f" % (x, y,res[x][y][0]))
outfile1 = file("result1.png", "w")
outfile2 = file("result2.png", "wb")
np.save(outfile1, res)
np.save(outfile2, res)
outfile1.close()
outfile2.close()
