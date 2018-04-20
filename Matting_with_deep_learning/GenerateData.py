import numpy as np
import Generate
a = Generate.generate1()
for i in range(100):
    print i
    np.save("data/{:0>3d}.npy".format(i),np.stack([a.next() for x in range(10000)]))

