import sys
import pdb
import json
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

range_max = int(sys.argv[1])
range_min = -1 * range_max

range_x = np.array([x/10 for x in range(10 * range_min * 2, 10 * range_max * 2)])
range_y = np.array([sigmoid(x) for x in range_x])

sigmoid_x = np.array([x/10 for x in range(10 * range_min, 10 * range_max)])
sigmoid_y = np.array([sigmoid(x) for x in sigmoid_x])

weight_dict = {}

for deg in range(3, 10, 2):
    print(deg)
    z = np.polyfit(sigmoid_x, sigmoid_y, deg)
    poly = np.poly1d(z)
    print(poly)

    weight_dict[deg] = z.tolist()

    # plt
    plot_y = poly(range_x)
    plt.plot(range_x, plot_y, label='poly%s' % deg)

plt.plot(range_x, range_y, label='sigmoid')

# plt.axis([range_min, range_max, -0.25, 1.25])
plt.axis([range_min*2, range_max*2, -1, 3])
plt.legend()
plt.savefig('./plot/sigmoid_%s.png' % (range_max))

with open('./poly_sigmoid_weights/weights_%s.out' % range_max ,'w') as outfile:
    json.dump(weight_dict, outfile, indent = 2)
