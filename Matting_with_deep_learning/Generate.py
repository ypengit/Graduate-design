import os 
import numpy as np
import cv2
import random

path = '/tmp/deep_matting/'
train_image_path  = path + 'input_training_lowres'
trimap_image_path = path + 'trimap_training_lowres/Trimap1'
alpha_image_path  = path + 'gt_training_lowres'

width = 20
height = 20

def file_names(file_dir):
    file_paths = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_paths.append(root + '/' + file)
    return file_paths

def file_content(filename):
    return cv2.imread(filename)

def files(file_dir):
    return [file_content(p) for p in file_names(file_dir)]

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

def get_block(index,pos):
    x = pos['x'] + width/2
    y = pos['y'] + height/2
    return np.lib.pad(data['train'][index], ((width/2,width/2),(height/2,height/2),(0,0)), 'constant', constant_values=(-255))[x-width/2:x+width/2,y-height/2:y+height/2]


def real_alpha(index, pos):
    return data['alpha'][index][pos['x']][pos['y']][0]/255


def cal_alpha(F, B, I):
    alpha = sum((I-B)*(F-B)) / (sum(F-B)**2 + 0.00001)
    if alpha > 1:
        return 1
    elif alpha < 0:
        return 0
    else:
        return alpha

def generate():
    data_pics = {}

    # pick one picture and get the shape
    index = random.randrange(num)
    shape = data['train'][index].shape
    data_pics['shape'] = shape

    # generate random pos according the shape of pictures
    data_pics['pos_I'] = rand_pos(shape, 'I', index)
    data_pics['pos_F'] = rand_pos(shape, 'F', index)
    data_pics['pos_B'] = rand_pos(shape, 'B', index)

    # get the block with pos
    data_pics['F'] = get_block(index, data_pics['pos_F'])
    data_pics['B'] = get_block(index, data_pics['pos_B'])
    data_pics['I'] = get_block(index, data_pics['pos_I'])

    # slice real F,B,I with pos
    F = data['train'][index][data_pics['pos_F']['x']][data_pics['pos_F']['y']]/255.0
    B = data['train'][index][data_pics['pos_B']['x']][data_pics['pos_B']['y']]/255.0
    I = data['train'][index][data_pics['pos_I']['x']][data_pics['pos_I']['y']]/255.0

    # calculate the difference between real_alpha and cal_alpha
    realalpha = real_alpha(index, data_pics['pos_I'])
    calalpha  = cal_alpha(F, B, I)
    data_pics['alpha_diff'] = realalpha - calalpha
    return data_pics

def next(n):
    ret = []
    for _ in range(n):
        ret.append(generate());
    return ret

