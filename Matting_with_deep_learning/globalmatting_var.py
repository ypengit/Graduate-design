import os
import glob
t = '/disk3/Graduate-design/source/alphamatting/var'
if not os.path.exists(t):
    os.mkdir(t)
path = '/disk3/Graduate-design/source/alphamatting/var/GlobalMattingResult/'
if not os.path.exists(path):
    os.mkdir(path)
if not os.path.exists(path):
    os.mkdir(path)
def generate():
    rgb_f = glob.glob(t + '/input_lowres/*.png')
    trimap_f = glob.glob(t + '/trimap_lowres/Trimap1/*.png')
    rgb_f.sort()
    trimap_f.sort()
    for i in range(len(rgb_f)):
        print rgb_f[i]
        name = rgb_f[i].split('/')[-1]
        res = t + '/GlobalMattingResult/' + name
        os.system("./GlobalMatting '" + rgb_f[i] + "' '" + trimap_f[i] + "' '" + res + "' '" + '' + "'" )
generate()
