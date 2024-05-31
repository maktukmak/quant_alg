import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from src.utils import float_grid



# Setup a plot such that only the bottom spine is shown
def setup(ax):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', width=0.75)
    ax.tick_params(which='minor', length=2.5)
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0, 1)
    ax.patch.set_alpha(0.0)


plt.figure(figsize=(8, 3))
n = 3

grid_nouni = np.sort(np.random.rand(128))
# Index Locator
ax = plt.subplot(n, 1, 1)
setup(ax)
ax.xaxis.set_major_locator(ticker.LinearLocator(3))
ax.xaxis.set_minor_locator(ticker.FixedLocator(grid_nouni))
ax.text(0.0, 0.1, "Non-Uniform grid",
        fontsize=14, transform=ax.transAxes)

grid_uni = np.linspace(0, 1, 128)
# Linear Locator
ax = plt.subplot(n, 1, 2)
setup(ax)
ax.xaxis.set_major_locator(ticker.LinearLocator(3))
ax.xaxis.set_minor_locator(ticker.FixedLocator(grid_uni))
ax.text(0.0, 0.1, "Uniform grid",
        fontsize=14, transform=ax.transAxes)

grid_fp = float_grid(E=4, M=3, b=7, special=0)
# Fixed Locator
ax = plt.subplot(n, 1, 3)
setup(ax)
majors = grid_fp
ax.xaxis.set_minor_locator(ticker.FixedLocator(grid_fp))
ax.xaxis.set_major_locator(ticker.LinearLocator(3))
ax.text(0.0, 0.1, "Floating point grid", fontsize=14,
        transform=ax.transAxes)





# Push the top of the top axes outside the figure because we only show the
# bottom spine.
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=1.05)
plt.savefig('grid.jpg', bbox_inches='tight')
plt.show()