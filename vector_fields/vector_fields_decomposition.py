import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

# Set limits and number of points in grid
y, x = np.mgrid[10:-10:100j, 10:-10:100j]

p1 = -y/50 + x/50
p2 = x/50
p3 = -y/50
def PlotVectorField(ax, p, x, y):
        dy, dx = np.gradient(p, np.diff(y[:2, 0]), np.diff(x[0, :2]))
        ax.streamplot(x, y, dx, dy, linewidth=100*np.hypot(dx, dy),
                      color='r', density=1.2)
        cont = ax.contour(x, y, p, cmap='gist_earth', vmin=p.min(), vmax=p.max())
        labels = ax.clabel(cont)
        plt.setp(labels, path_effects=[withStroke(linewidth=8, foreground='w')])
        ax.plot([-10,10],[0,0],'-k',linewidth=5)
        ax.plot(0.0,0.0,'ok',markersize=10)
        ax.set_ylim([-10,10])
        ax.set_xlim([-10,10])

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
fig.patch.set_facecolor('white')

PlotVectorField(ax1, p1, x, y)
ax1.set(aspect=1, title='Regular Vector Field')
PlotVectorField(ax2, p2, x, y)
ax2.set(aspect=1, title='Tangential Field')
PlotVectorField(ax3, p3, x, y)
ax3.set(aspect=1, title='Orthogonal Field')

plt.show()
