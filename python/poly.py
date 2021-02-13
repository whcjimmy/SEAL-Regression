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

sigmoid_x = [x/10 for x in range(10 * range_min, 10 * range_max)]
sigmoid_y = [sigmoid(x) for x in sigmoid_x]

data = {}

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
