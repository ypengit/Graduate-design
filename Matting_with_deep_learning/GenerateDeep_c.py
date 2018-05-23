import os 
import cProfile
import numpy as np
import cv2
import pandas as pd
import random
import glob
path = '/disk3/Graduate-design/source/newdataset/'
train_image_path  = path + 'final_rgb'
trimap_image_path = path + 'final_trimap'
alpha_image_path  = path + 'final_alpha'
fg_image_path = path + 'final_fg'
bg_image_path = path + 'final_bg'
global_image_path = path + 'GlobalMattingResult'
shared_image_path = path + 'SharedMattingResult'
knn_image_path = path + 'KNNMattingResult'

rgb_f = glob.glob(train_image_path + '/*.png')

dirs = ['/disk3/Graduate-design/data/newdataset/rgb/',
        '/disk3/Graduate-design/data/newdataset/trimap/',
        '/disk3/Graduate-design/data/newdataset/alpha/',
        '/disk3/Graduate-design/data/newdataset/fg/',
        '/disk3/Graduate-design/data/newdataset/bg/',]
for x in dirs:
    if not os.path.exists(x):
        os.mkdir(x)

width = 320
height = 320
num = len(rgb_f)
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
    count = 15500
    dirs = np.load('/disk3/Graduate-design/source/newdataset/names.npy')
    while True:
        i = random.randrange(len(dirs))
        pre = '/'.join(rgb_f[i].split('/')[:-2])
        # suf = rgb_f[i].split('/')[-1]
        suf = dirs[i]
        print suf
        rgb = cv2.imread(rgb_f[i])
        if not os.path.exists(pre + '/final_alpha/' + suf):
            print pre + '/final_alpha/' + suf , ' not exists!'
        alpha = cv2.imread(pre + '/final_alpha/' + suf)
        bg = cv2.imread(pre + '/final_bg/' + suf)
        fg = cv2.imread(pre + '/final_fg/' + suf)
        trimap = cv2.imread(pre + '/final_trimap/' + suf)
        tmp = 0
        while tmp < 20:
            h, w, _ = rgb.shape
            x = random.randrange(w)
            y = random.randrange(h)
            if(x + width < w and y + height < h):
                if count > n-1:
                    return
                tmp += 1
                cv2.imwrite("/disk3/Graduate-design/data/newdataset/rgb/{:0>6}.png".format(count),rgb[y:y+height, x:x+width, :])
                cv2.imwrite("/disk3/Graduate-design/data/newdataset/fg/{:0>6}.png".format(count),fg[y:y+height, x:x+width, :])
                cv2.imwrite("/disk3/Graduate-design/data/newdataset/bg/{:0>6}.png".format(count),bg[y:y+height, x:x+width, :])
                cv2.imwrite("/disk3/Graduate-design/data/newdataset/trimap/{:0>6}.png".format(count),trimap[y:y+height, x:x+width, :])
                cv2.imwrite("/disk3/Graduate-design/data/newdataset/alpha/{:0>6}.png".format(count),alpha[y:y+height, x:x+width, :])
                count+=1
                if count % 1000 == 0:
                    print count

def main():
    generate(200000)

if __name__ == "__main__":
    main()
