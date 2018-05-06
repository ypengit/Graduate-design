import os 
import cProfile
import numpy as np
import cv2
import pandas as pd
import random
import glob
path = '/tmp/deep_matting/'
train_image_path  = path + 'input_training_lowres'
trimap_image_path = path + 'trimap_training_lowres/Trimap1'
alpha_image_path  = path + 'gt_training_lowres'
width = 320
height = 320
def file_names(file_dir):
    file_paths = glob.glob(file_dir + "/*.png")
    file_paths.sort()
    return file_paths
def file_content(filename,t):
    return cv2.imread(filename)
def files(file_dir,t='others'):
    return [file_content(p,t) for p in file_names(file_dir)]
data = {}
data['train'] = files(train_image_path)
data['trimap']= files(trimap_image_path)
data['alpha'] = files(alpha_image_path)
num = len(data['train'])
def pos_f(index, val):
    while True:
        x = random.randrange(data['trimap'][index].shape[0])
        y = random.randrange(data['trimap'][index].shape[1])
        if(np.equal(data['trimap'][index][x][y], np.array(val)).all()):
            return x,y
def rand_pos(shape, types, index):
    pos = {}
    if(types == 'I'):
        pos['x'], pos['y'] = pos_f(index, [128, 128, 128])
    if(types == 'F'):
        pos['x'], pos['y'] = pos_f(index, [255, 255, 255])
    if(types == 'B'):
        pos['x'], pos['y'] = pos_f(index, [0, 0, 0])
    return pos
def get_val(index,pos):
    x = pos['x']
    y = pos['y']
    return data['train'][index][x][y]
def get_block(index,pos):
    x = pos['x'] + width/2
    y = pos['y'] + height/2
    return np.lib.pad(data['train'][index], ((width/2,width/2),(height/2,height/2),(0,0)), 'constant', constant_values=(-255))[x-width/2:x+width/2+1,y-height/2:y+height/2+1]
def real_alpha(index, pos):
    alpha = data['alpha'][index][pos['x']][pos['y']][0]/255
    return alpha

def cal_alpha_f(F, B, I):
    alpha =  np.sum(((I-B)*(F-B)), 2)/np.sum(((F-B)**2 + 0.0001), 2)
    alpha = np.where((alpha>1),np.ones_like(alpha), alpha)
    alpha = np.where((alpha<0),np.zeros_like(alpha), alpha)
    return np.expand_dims(alpha, -1) * 255.0
def generate(n):
    # pick one picture and get the shape
    count = 30000
    while True:
        idx = random.randrange(num)
        h, w, _ = data['train'][idx].shape
        while True:
            x1 = random.randrange(w)
            y1 = random.randrange(h)
            if(x1 + width < w and y1 + height < h):
                break
        while True:
            x2 = random.randrange(w)
            y2 = random.randrange(h)
            if(x2 + width < w and y2 + height < h):
                break
        while True:
            x3 = random.randrange(w)
            y3 = random.randrange(h)
            if(x3 + width < w and y3 + height < h):
                break
        if count > n-1:
            return
        data_pics = {}

        # FBI
        data_pics['F'] = data['train'][idx][y1:y1+height,x1:x1+width,:]
        data_pics['B'] = data['train'][idx][y2:y2+height,x2:x2+width,:]
        data_pics['I'] = data['train'][idx][y3:y3+height,x3:x3+width,:]

        # trimap_FBI
        data_pics['trimap_f'] = data['trimap'][idx][y1:y1+height,x1:x1+width,:]
        data_pics['trimap_b'] = data['trimap'][idx][y2:y2+height,x2:x2+width,:]
        data_pics['trimap_i'] = data['trimap'][idx][y3:y3+height,x3:x3+width,:]

        # distance between F and I, B and I
        data_pics['distance_fi'] = np.full([height, width, 1],pow((pow((x1 - x3), 2) + pow((y1 - y3), 2)), 0.5))
        data_pics['distance_bi'] = np.full([height, width, 1],pow((pow((x2 - x3), 2) + pow((y2 - y3), 2)), 0.5))

        # alpha
        data_pics['real_alpha'] = data['alpha'][idx][y3:y3+height,x3:x3+width,:]
        data_pics['cal_alpha'] = cal_alpha_f(data_pics['F'], data_pics['B'], data_pics['I'])
        data_pics['diff_alpha'] = data_pics['real_alpha'] - data_pics['cal_alpha']

        cv2.imwrite("/disk3/Graduate-design/data/method2/F/{:0>6}.png".format(count),data_pics['F'])
        cv2.imwrite("/disk3/Graduate-design/data/method2/B/{:0>6}.png".format(count),data_pics['B'])
        cv2.imwrite("/disk3/Graduate-design/data/method2/I/{:0>6}.png".format(count),data_pics['I'])

        cv2.imwrite("/disk3/Graduate-design/data/method2/trimap_f/{:0>6}.png".format(count),data_pics['trimap_f'])
        cv2.imwrite("/disk3/Graduate-design/data/method2/trimap_b/{:0>6}.png".format(count),data_pics['trimap_b'])
        cv2.imwrite("/disk3/Graduate-design/data/method2/trimap_i/{:0>6}.png".format(count),data_pics['trimap_i'])

        cv2.imwrite("/disk3/Graduate-design/data/method2/real_alpha/{:0>6}.png".format(count),data_pics['real_alpha'])
        cv2.imwrite("/disk3/Graduate-design/data/method2/cal_alpha/{:0>6}.png".format(count),data_pics['cal_alpha'])

        np.save("/disk3/Graduate-design/data/method2/distance_fi/{:0>6}.npy".format(count),data_pics['distance_fi'])
        np.save("/disk3/Graduate-design/data/method2/distance_bi/{:0>6}.npy".format(count),data_pics['distance_bi'])

        np.save("/disk3/Graduate-design/data/method2/diff_alpha/{:0>6}.npy".format(count),data_pics['diff_alpha'])
        count+=1
        if count % 100 == 0:
            print count

def main():
    generate(200000)

if __name__ == "__main__":
    main()
