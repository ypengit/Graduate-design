import numpy as np
import GenerateTest
a = GenerateTest.generate1()
for i in range(1):
    print i
    np.save("/disk3/Graduate-design/test/{:0>3d}.npy".format(i),np.stack([a.next() for x in range(1000)]))

