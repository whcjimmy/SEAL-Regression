import pdb
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def integrand(x, poly):
    return poly(x) - sigmoid(x)


range_min = -15
range_max = 15

sigmoid_x = np.array([x/10 for x in range(10 * range_min, 10 * range_max)])
sigmoid_y = np.array([sigmoid(x) for x in sigmoid_x])
plot_x = np.array([x/10 for x in range(10 * range_min * 2, 10 * range_max * 2)])
plot_y = np.array([sigmoid(x) for x in plot_x])

data = {}

# Original Version
for deg in range(3, 10, 2):
    print(deg)
    z = np.polyfit(sigmoid_x, sigmoid_y, deg)
    poly = np.poly1d(z)
    print(poly)

    data[deg] = z.tolist()

    # plt
    poly_y = poly(plot_x)
    plt.plot(plot_x, poly_y, label='poly%s' % deg)

    # Errors 
    I = quad(integrand, -7, 7, args = (poly))
    print('Erros : ', I)

plt.plot(plot_x, plot_y, label='sigmoid')

# plt.axis([range_min, range_max, -0.25, 1.25])
plt.axis([range_min*2, range_max*2, -1, 3])
plt.legend()
plt.savefig('./plot_sigmoid_%s_%s.png' % (range_min, range_max))

with open('weights.out' ,'w') as outfile:
    json.dump(data, outfile, indent = 2)
