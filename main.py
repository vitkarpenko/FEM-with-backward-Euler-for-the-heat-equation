from config import *


def triangulate_rectangle(x_min, x_max, y_min, y_max, grid_density):
    points = np.dstack((np.linspace(x_min, x_max, grid_density), np.full(grid_density, y_min, dtype='float')))[0]
    for i in range(1, grid_density):
        c = i / (grid_density - 1)
        points = np.concatenate((points, np.dstack((np.linspace(x_min, x_max, grid_density), np.full(grid_density, y_min + (y_max - y_min) * c, dtype='float')))[0]), axis=0)
    tri = scipy.spatial.Delaunay(points)
    plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()


triangulate_rectangle(x_min, x_max, y_min, y_max, grid_density)
