import pdb
import numpy as np
import math
import matplotlib.pyplot as plt


x = [i for i in range(1, 31, 1)]
y_1 = [0.707317 for i in range(30)]
y_2 = [0.263415, 0.253659, 0.253659, 0.265415, 0.268293, 0.287805, 
       0.302439, 0.331707, 0.336585, 0.360976, 0.380488, 0.390244, 
       0.429268, 0.458537, 0.482927, 0.512195, 0.517073, 0.526829, 
       0.541463, 0.546341, 0.565854, 0.595122, 0.604878, 0.629268,
       0.648780, 0.653659, 0.658537, 0.673171, 0.678049, 0.697561]

plt.plot(x, y_1, label = 'plaintext domain')
plt.plot(x, y_2, label = 'ciphertext domain')

plt.legend()

plt.savefig('./plot_accuracy_curve.png')




