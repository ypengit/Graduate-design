import os
import random
import glob
t = '/disk3/Graduate-design/source/newdataset'
path = '/disk3/Graduate-design/source/newdataset/GlobalMattingResult/'
if not os.path.exists(path):
    os.mkdir(path)
def generate():
    rgb_f = glob.glob(t + '/final_rgb/*.png')
    trimap_f = glob.glob(t + '/final_trimap/*.png')
    alpha_f = glob.glob(t + '/final_alpha/*.png')
    rgb_f.sort()
    alpha_f.sort()
    trimap_f.sort()
    random.shuffle(rgb_f)
    for i in range(len(rgb_f)):
        print rgb_f[i]
        name = rgb_f[i].split('/')[-1]
        res = t + '/GlobalMattingResult/' + name
        trimap_n = t + '/final_trimap/' + name
        alpha_n = t + '/final_alpha/' + name
        if os.path.exists(res):
            print 'skip the image ' + res
            continue
        os.system("./GlobalMatting '" + rgb_f[i] + "' '" + trimap_n + "' '" + res + "' '" + alpha_n + "'" )
generate()
