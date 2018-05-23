from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import imageio
import cv2
import glob
# load the image and convert it to a floating point data type

def generate():
    bg_f = glob.glob('/disk3/Graduate-design/newdataset/Training_set/final_fg/*.png')
    # loop over the number of segments
    for v in bg_f:
        # apply SLIC and extract (approximately) the supplied number
        # of segments
        image = cv2.imread(v) / 255.0
        segments = slic(image, n_segments = numSegments, sigma = 5)
        # show the output of SLIC
        cv2.imwrite('{}.png'.format(numSegments), 255.0 * mark_boundaries(image, segments))
