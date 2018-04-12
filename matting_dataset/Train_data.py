# -*-coding:utf-8-*- 
import numpy as np
import random
import cv2
# import tensorflow as tf


class data(object):
    def generatePos(self, mask):
        return [[i,j] for i in range(0,mask.shape[0]) for j in range(0, mask.shape[1]) if mask[i][j].all() == False]

    def get(self,pos):
        if pos[0]<11 or pos[0]>self.img.shape[0]-11:
            pos[0] = random.randrange(11,self.img.shape[0]-11)
        if pos[1]<11 or pos[1]>self.img.shape[1]-11:
            pos[1] = random.randrange(11,self.img.shape[1]-11)
        return self.img[pos[0] - 10:pos[0] + 11,pos[1] - 10:pos[1] + 11]

    def sum_(self,a):
        return sum(sum(sum(a)))

    def calcAlpha(self, f_, b_, i_):
        alpha = self.sum_((i_-b_)*(f_-b_))/(self.sum_((f_-b_)*(f_-b_)) + 0.001)
        if alpha>1:
            return 1;
        elif alpha<0:
            return 0;
        else:
            return alpha;


        
    def getBlocks(self, n, f, b, u, uncertainalpha):
        f1 = random.sample(f, n)
        b1 = random.sample(b, n)
        u  = random.sample(zip(u,uncertainalpha), n)
        u1 = list(np.array(u)[:,0])
        u2 = np.array(u)[:,1]
        resx_ = []
        resy_ = []
        for i in range(0, n):
            F = np.array(self.get(f1[i]))
            B = np.array(self.get(b1[i]))
            I = np.array(self.get(u1[i]))
            x_ =[list(F[x][y]) + list(B[x][y]) + list(I[x][y]) for x in range(20) for y in range(20)]
            x_ = np.reshape(x_, (20, 20, 9))
            y_ = u2[i][0] - self.calcAlpha(F, B, I);
            resx_.append(x_)
            resy_.append([y_])
        return np.array(resx_),np.array(resy_)
        


            

            # 下一步的工作是构建出输入，其输入应该是20*20*3的矩阵，输出为真实alpha和计算alpha之间的差值
            # 主要是构建训练集

        

    img = cv2.imread("./input_training_lowres/GT01.png")
    trimap = cv2.imread("./trimap_training_lowres/Trimap1/GT01.png")
    gt = cv2.imread("./gt_training_lowres/GT01.png")
    uncertainMask = ~((trimap ==  1) ^ (trimap ==  255))
    frontMask = trimap == 1
    backMask = trimap == 255
    uncertainMask = trimap == 128
    frontPos = generatePos(object, frontMask)
    backPos = generatePos(object, backMask)
    uncertainPos = generatePos(object, uncertainMask)
    uncertainalpha = [[gt[i][j][0]/255.0] for i in range(0,uncertainMask.shape[0]) for j in range(0,uncertainMask.shape[1]) if uncertainMask[i][j].all() == False]
        
    def next_batch(self, n):
        return self.getBlocks(n, self.frontPos, self.backPos, self.uncertainPos, self.uncertainalpha)



train_data = data()
x_,y_ = train_data.next_batch(10)





#for i in range(uncertain.shape[0]):
#    for j in range(uncertain.shape[1]):
#        if uncertain[i][j].all() == True:
#            t = []
#            t.append(img[i][j])
#            t.append([i, j])
            
