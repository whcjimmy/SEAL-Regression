import pdb
import numpy as np
import math
import matplotlib.pyplot as plt

range_min = -15
range_max = 15

e_x = np.array([x/10 for x in range(10 * range_min, 10 * range_max)])
e_y = np.array([np.exp(x) for x in e_x])

for deg in range(2, 30, 8):
    print(deg)
    z = []
    for i in range(deg, 0, -1):
        z.append(math.factorial(i))
    z = [1/x for x in z]
    z.append(1.0)
    z = np.array(z)

    poly = np.poly1d(z)
    print(poly)
    poly_y = poly(e_x)
    plt.plot(e_x, poly_y, label='%sth Taylor Polynomial' % deg)


plt.plot(e_x, e_y, label='e^x')

plt.axis([-15, 5, 0, 10])
plt.legend()

plt.savefig('./exp.png')

