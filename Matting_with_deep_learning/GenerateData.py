import numpy as np
import Generate
a = Generate.generate1()
for i in range(0,1000,1):
    print i
    np.save("/disk3/Graduate-design/data/{:0>3d}.npy".format(i),np.stack([a.next() for x in range(1000)]))

