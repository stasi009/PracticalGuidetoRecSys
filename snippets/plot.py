
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

mean = 0
std = 1

xmax = 5
step = 0.01
xs = np.arange(-xmax, xmax, step)
ys = norm.pdf(xs, mean, std)

plt.plot(xs, ys)

sl = 1.96
z = 2.5
x_points = [(sl, 'r'), (z, 'b')]
# x_points = [(sl, 'r')]

for (x, color) in x_points:
    for xx in [-x, x]:
        plt.axvline(x=xx, ymax=0.5, color=color, ls='--', lw=2)


for (fx,color) in [(z, 'b')]:
    fill_x = np.arange(fx,xmax,step)
    plt.fill_between(fill_x,norm.pdf(fill_x),color=color)

    fill_x = np.arange(-xmax,-fx,step)
    plt.fill_between(fill_x,norm.pdf(fill_x),color=color)

plt.ylim(0, 0.5)
plt.grid()
plt.show()
