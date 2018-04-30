import wget
import os
import zipfile

url = ['http://www.alphamatting.com/datasets/zip/gt_training_lowres.zip',
       'http://www.alphamatting.com/datasets/zip/trimap_training_lowres.zip',
       'http://www.alphamatting.com/datasets/zip/input_training_lowres.zip',
       'http://alphamatting.com/datasets/zip/input_lowres.zip',
       'http://alphamatting.com/datasets/zip/trimap_lowres.zip'
       ]
path = '/tmp/deep_matting/'
files = ['gt_training_lowres.zip', 'trimap_training_lowres.zip', 'input_training_lowres.zip', 'input_lowres.zip', 'trimap_lowres.zip']

def download(url, filename, path=path):
    if not os.path.exists(path):
        os.mkdir(path)
    p = path + filename
    if not os.path.exists(p):
        print('Downloading %20s ..... ' % filename)
        wget.download(url=url, out=p)
        print('Finished ! \n')

def unzip(filename):
    p = path + ''.join(filename.split('.')[:-1]) 
    filename = path + filename
    with zipfile.ZipFile(filename,'r') as zip_ref:
        zip_ref.extractall(p)
    print('Zipping %20s , finished! \n' % filename)

def main():
    for u, f in zip(url, files):
        download(u, f,)
        unzip(f)
if __name__ == '__main__':
    main()
