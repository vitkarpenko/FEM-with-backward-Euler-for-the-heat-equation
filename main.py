from config import *


def generate_nodes(x_min, x_max, y_min, y_max, grid_density):
    '''Разделяет пластинку на равномерную прямоугольную сетку.
    grid_density шагов на каждой стороне.'''
    nodes = np.dstack((np.linspace(x_min, x_max, grid_density),
                        np.full(grid_density, y_min, dtype='float')))[0]
    for i in range(1, grid_density):
        c = i / (grid_density - 1)
        nodes = np.concatenate((nodes, np.dstack((np.linspace(x_min, x_max, grid_density),
                                                    np.full(grid_density, y_min + (y_max - y_min) * c,
                                                                dtype='float')))[0]), axis=0)
    return nodes


def triangulate_rectangle(nodes):
    '''Триангулирует пластинку по заданным узлам методом Делоне.'''
    tri = scipy.spatial.Delaunay(nodes)
    return tri.simplices.copy()


# dir_nodes - номера узлов с условием Дирихле, в данном случае внешние
# ind_nodes - номера свободных, т.е. в данном случае внутренних узлов
nodes_coords = generate_nodes(x_min, x_max, y_min, y_max, grid_density)
dir_nodes = np.array([i + grid_density * j for j in range(grid_density) for i in range(grid_density)
                                                                if i == 0 or i == grid_density - 1
                                                                or j == 0 or j == grid_density - 1])
ind_nodes = np.array([x for x in range(grid_density ** 2) if x not in dir_nodes])

triangles = triangulate_rectangle(nodes_coords)
