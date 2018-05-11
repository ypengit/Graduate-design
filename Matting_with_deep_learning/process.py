import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from sklearn.cluster import KMeans

idx = 26
rgb = cv2.imread("/tmp/deep_matting/test/input_training_lowres/GT{:0>2}.png".format(idx+1))

trimap = cv2.imread("/tmp/deep_matting/test/trimap_training_lowres/Trimap1/GT{:0>2}.png".format(idx+1))[...,0:1]

rgb_t = np.concatenate([rgb,trimap],2)

data = []

for idx_y,y in enumerate(rgb_t):
    for idx_x,x in enumerate(y):
        data_pics = []
        for v in x:
            data_pics.append(v)
        data_pics.append(idx_x)
        data_pics.append(idx_y)
        data.append(data_pics)

data = pd.DataFrame(data,columns=['g','b','r','x','y','trimap'])

kmeans = KMeans(n_clusters=100, random_state=0).fit(data)

data['sum'] = data['r'] + data['g'] + data['b']

data.insert(7, column='type', value=kmeans.labels_)

color_type = data.groupby(['type']).mean()

color_type  = np.floor(color_type[['g','b','r']])

data = data.join(color_type,how='right',on=['type'], rsuffix='_c')

rgb_c = np.copy(rgb)

colors = data[['g_c','b_c','r_c']]
for idx_y,y in enumerate(rgb_c):
    for idx_x,x in enumerate(y):
        rgb_c[idx_y][idx_x] = colors.loc[idx_y * rgb_c.shape[1] + idx_x]

cv2.imwrite("res/{:0>2}.png".format(idx+1),rgb_c)
