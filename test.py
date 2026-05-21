import numpy as np, matplotlib.pyplot as plt
arr = np.load("C://Users//JIANG//Desktop//dataset//FD_1mm_sharp//full_1mm_sharp//L067//L067_FD_1_SHARP_1.CT.0002.0008.2016.01.21.18.11.40.977560.404629183.npy")
plt.imshow(arr.reshape(512, 512), cmap='gray')
plt.savefig('check.png')