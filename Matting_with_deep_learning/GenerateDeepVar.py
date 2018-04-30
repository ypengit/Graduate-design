import os 
import cProfile
import numpy as np
import cv2
import pandas as pd
import random
import glob
path = '/tmp/deep_matting/var/'
train_image_path  = path + 'input_lowres'
trimap_image_path = path + 'trimap_lowres/Trimap1'
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
            cv2.imwrite("/disk3/Graduate-design/var/rgb/{:0>6}.png".format(count),data_pics['rgb'])
            cv2.imwrite("/disk3/Graduate-design/var/trimap/{:0>6}.png".format(count),data_pics['trimap'])
            count+=1
            if count % 1000 == 0:
                print count

def main():
    generate(20000)

if __name__ == "__main__":
    main()
