import pdb
import random
import cv2
import numpy as np
import pandas as pd
import matlplotlib.pyplot as plt
from sklearn.cluster import KMeans

cluster_num = 100

def random_color():
    rgbl = [256, 256, 256]
    rand = np.random.rand(1,3)
    return np.floor(rand * rgbl)

for idx in range(27):
    rgb = cv2.imread("/tmp/deep_learning/")
    trimap = cv2.imread("/tmp/deep_learning/")
    rgb_t = np.con
    data = []
    for idx_y,y in enumerate(rgb_t):
        for idx_x,x in enumerate(y):
            data_pics = []
            for v in x:
                data_pics.append(v)
                data_pics.append(idx_x)
                data_pics.append(idx_y)
                data.append(data_pics)
    kmeans = KMeans(n_cluster=100, random_state=0).fit(data)
    pdb.set_trace()
    data = np.array(data)
    data = pd.DataFrame(data, columns=['g', 'b', 'r', 'x', 'y', 'type'])
    data['sum'] = data['r'] + data['g'] + data['b']
    colors = np.array([random_color() for _ in range(cluster_num)])

    for idx_y,y in enumerate(rgb_c):
        for idx_x,x in enumerate(y):
            rgb_c[idx_y][idx_x] = colors[labels[idx_y * rgb_c.shape[1] + idx_x]]


            
