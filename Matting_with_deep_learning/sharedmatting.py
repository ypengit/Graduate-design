import os
import glob
t = '/disk3/Graduate-design/source/newdataset'
path = t + '/SharedMattingResult/'
if not os.path.exists(path):
    os.mkdir(path)
def generate():
    rgb_f = glob.glob(t + '/final_rgb/*.png')
    trimap_f = glob.glob(t + '/final_trimap/*.png')
    alpha_f = glob.glob(t + '/final_alpha/*.png')
    rgb_f.sort()
    alpha_f.sort()
    trimap_f.sort()
    for i in range(len(rgb_f)):
        print rgb_f[i]
        name = rgb_f[i].split('/')[-1]
        res = t + '/SharedMattingResult/' + name
        os.system("./SharedMatting '" + rgb_f[i] + "' '" + trimap_f[i] + "' '" + res + "' '" + alpha_f[i] + "'" )
generate()
