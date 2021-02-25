import pdb
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def integrand(x, poly):
    return poly(x) - sigmoid(x)


range_min = -10
range_max = 10

sigmoid_x = np.array([x/10 for x in range(10 * range_min, 10 * range_max)])
sigmoid_y = np.array([sigmoid(x) for x in sigmoid_x])

data = {}

'''
x_1 = np.power(sigmoid_x, 1)
x_2 = np.power(sigmoid_x, 2)
x_3 = np.power(sigmoid_x, 3)
x_4 = np.power(sigmoid_x, 4)
x_8 = np.power(sigmoid_x, 8)
x_16 = np.power(sigmoid_x, 16)

# 2
A = np.vstack([x_2, x_1, np.ones(len(sigmoid_x))]).T
m, _, _, _ = np.linalg.lstsq(A, sigmoid_y, rcond = None)
print(m)
poly_y = [m[0]*a + m[1]*b + m[2] for a, b in zip(x_2, x_1)]
plt.plot(sigmoid_x, poly_y, label='poly_2')

# 3
A = np.vstack([x_3, x_2, x_1, np.ones(len(sigmoid_x))]).T
m, _, _, _ = np.linalg.lstsq(A, sigmoid_y, rcond = None)
print(m)
poly_y = [m[0]*a + m[1]*b + m[2]*c + m[3] for a, b, c in zip(x_3, x_2, x_1)]
plt.plot(sigmoid_x, poly_y, label='poly_3')

# 4
A = np.vstack([x_4, x_3, x_2, x_1, np.ones(len(sigmoid_x))]).T
m, _, _, _ = np.linalg.lstsq(A, sigmoid_y, rcond = None)
print(m)
poly_y = [m[0]*a + m[1]*b + m[2]*c + m[3]*d + m[4] for a, b, c, d in zip(x_4, x_3, x_2, x_1)]
plt.plot(sigmoid_x, poly_y, label='poly_4')
'''

# Original Version
for deg in range(3, 11, 2):
    print(deg)
    z = np.polyfit(sigmoid_x, sigmoid_y, deg)
    poly = np.poly1d(z)
    print(poly)

    data[deg] = z.tolist()

    # plt
    poly_y = poly(sigmoid_x)
    plt.plot(sigmoid_x, poly_y, label='poly%s' % deg)

    # Errors 
    I = quad(integrand, -7, 7, args = (poly))
    print('Erros : ', I)

plt.plot(sigmoid_x, sigmoid_y, label='sigmoid')
plt.legend()
plt.savefig('./poly.png')

with open('weights.out' ,'w') as outfile:
    json.dump(data, outfile, indent = 2)
