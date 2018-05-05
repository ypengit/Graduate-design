import numpy as np
def generate(idx):
    data_pics = {}
    x = np.array([255.0, 255.0, 255.0])

    rand = np.random.rand(1,3)
    F = np.floor(rand * x)

    rand = np.random.rand(1,3)
    B = np.floor(rand * x)

    rand = np.random.rand(1,3)
    I = np.floor(rand * x)

    data_pics['F'] = F
    data_pics['B'] = B
    data_pics['I'] = I

    cal_alpha = np.sum((I - B) * (F - B)) / (np.sum((F - B) * (F - B)) + 0.01)
    data_pics['cal_alpha'] = cal_alpha

    np.save('/disk3/Graduate-design/t/{:0>6}.npy'.format(idx),np.array([data_pics]))

def main():
    for i in range(100000):
        generate(i)

if __name__ == main():
    main()
