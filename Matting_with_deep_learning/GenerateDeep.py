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
processed_image_path = path + 'processed'
width = 320
height = 320
def file_names(file_dir):
    file_paths = glob.glob(file_dir + "/*.png")
    file_paths.sort()
    return file_paths
def file_content(filename,t):
    if t == 'alpha':
        return cv2.imread(filename,cv2.CV_8UC1)
    return cv2.imread(filename)
def files(file_dir,t='others'):
    return [file_content(p,t) for p in file_names(file_dir)]
data = {}
data['train'] = files(train_image_path)
data['trimap']= files(trimap_image_path)
data['alpha'] = files(alpha_image_path,'alpha')
data['processed'] = files(processed_image_path)
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

def cal_alpha(F, B, I):
    alpha =  sum((I-B)*(F-B))/sum((F-B)**2 + 0.0001)
    if alpha > 1:
        return 1
    if alpha < 0:
        return 0
    return alpha
def generate(n):
    # pick one picture and get the shape
    count = 0
    while True:
        idx = random.randrange(num)
        h, w, _ = data['train'][idx].shape
        x = random.randrange(w)
        y = random.randrange(h)
        if(x + width < w and y + height < h):
            if count > n-1:
                return
            data_pics = {}
            # data_pics['idx'] = idx
            # data_pics['x'] = x
            # data_pics['y'] = y
            data_pics['rgb'] = data['train'][idx][y:y+height,x:x+width,:]
            data_pics['trimap'] = data['trimap'][idx][y:y+height,x:x+width,:]
            data_pics['alpha'] = data['alpha'][idx][y:y+height,x:x+width]
            data_pics['processed'] = data['processed'][idx][y:y+height,x:x+width]
            cv2.imwrite("/disk3/Graduate-design/data/rgb/{:0>6}.png".format(count),data_pics['rgb'])
            cv2.imwrite("/disk3/Graduate-design/data/trimap/{:0>6}.png".format(count),data_pics['trimap'])
            cv2.imwrite("/disk3/Graduate-design/data/alpha/{:0>6}.png".format(count),data_pics['alpha'])
            cv2.imwrite("/disk3/Graduate-design/data/processed/{:0>6}.png".format(count),data_pics['processed'])
            count+=1
            if count % 1000 == 0:
                print count

def main():
    generate(200000)

if __name__ == "__main__":
    main()
