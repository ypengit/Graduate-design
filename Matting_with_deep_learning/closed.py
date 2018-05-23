import closed_form_matting
import os
import glob
import cv2
rgb_f = glob.glob('/disk3/Graduate-design/source/newdataset/final_rgb/*.png')
trimap_f = glob.glob('final_trimap/*.png')
t = '/disk3/Graduate-design/source/newdataset/Closed/'
if not os.path.exists(t):
    os.mkdir(t)
for i in range(len(rgb_f)):
    print i
    image = cv2.imread(rgb_f[i])
    trimap = cv2.imread('/disk3/Graduate-design/source/newdataset/final_trimap/'+rgb_f[i].split('/')[-1])[...,0]
    alpha = closed_form_matting.closed_form_matting_with_trimap(image, trimap)
    cv2.imwrite(t + rgb_f[i].split('/')[-1],alpha)
