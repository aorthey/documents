import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

# Set limits and number of points in grid
y, x = np.mgrid[10:-10:100j, 10:-10:100j]

#alpha_obstacle, a_obstacle, b_obstacle = 1.0, 1e3, 2e3
#p = -alpha_obstacle * np.exp(-(x**2 / a_obstacle + y**2 / b_obstacle))
#p1 = -np.exp(-(x**2/1000 + y**2/1000))
#p2 = -np.exp(-(x**2)/1000)
#p3 = y/50

p1 = (x**2/1000 + y**2/1000)
p3 = -(x**2/1000 + y**2/1000)
def PlotVectorField(ax, p, x, y):
        #ax.streamplot(x, y, dx, dy, linewidth=500*np.hypot(dx, dy),
                      #color=p, density=1.2, cmap='gist_earth')
        dy, dx = np.gradient(p, np.diff(y[:2, 0]), np.diff(x[0, :2]))
        #ax.streamplot(x, y, dx, dy, linewidth=100*np.hypot(dx, dy),
                      #color=p, density=1.2, cmap=plt.cm.autumn)
        ax.streamplot(x, y, dx, dy, linewidth=100*np.hypot(dx, dy),
                      color='r', density=1.2)
        cont = ax.contour(x, y, p, cmap='gist_earth', vmin=p.min(), vmax=p.max())
        labels = ax.clabel(cont)
        plt.setp(labels, path_effects=[withStroke(linewidth=8, foreground='w')])
        ax.plot([-10,10],[3,-3],'-k',linewidth=5)
        ax.plot(0.0,0.0,'ok',markersize=10)
        ax.set_ylim([-10,10])
        ax.set_xlim([-10,10])

fig, (ax1,ax3) = plt.subplots(1,2)
fig.patch.set_facecolor('white')

PlotVectorField(ax1, p1, x, y)
ax1.set(aspect=1, title='Unstable Singular Point Vector Field\n(Source)')
PlotVectorField(ax3, p3, x, y)
ax3.set(aspect=1, title='Stable Singular Point Vector Field\n(Sink)')

plt.show()

