from config import *


def generate_nodes(x_min, x_max, y_min, y_max, grid_density):
    '''Разделяет пластинку на равномерную прямоугольную сетку.
    grid_density шагов на каждой стороне.
    '''
    nodes = np.dstack((np.linspace(x_min, x_max, grid_density),
                        np.full(grid_density, y_max, dtype='float')))[0]
    for i in range(1, grid_density):
        c = i / (grid_density - 1)
        nodes = np.concatenate((nodes, np.dstack((np.linspace(x_min, x_max, grid_density),
                                                    np.full(grid_density, y_max - (y_max - y_min) * c,
                                                                dtype='float')))[0]), axis=0)
    return nodes


def triangulate_rectangle(nodes):
    '''Триангулирует пластинку по заданным узлам методом Делоне.
    '''
    tri = scipy.spatial.Delaunay(nodes)
    return tri.simplices.copy()


def N(a, K, x, y):
    '''Узловая базисная функция в треугольнике с номером K, в узле с локальным номером a
    в точке (x, y). Вычисляется с помощью перехода к опорному элементу
    с узлами (0, 0), (0, 1), (1, 0).
    '''
    # определяем координаты треугольника
    x1, y1, x2, y2, x3, y3 = nodes_coords[triangles[K]].flatten()
    # вычисляем координаты u, v в опорном элементе по x, y с помощью афинного преобразования
    u, v = (np.linalg.inv(np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]]))
               @ np.array([x - x1, y - y1]))
    # возвращаем результат в зависимости от номера базисной функции
    if a == 0:
        return 1 - u - v
    elif a == 1:
        return u
    elif a == 2:
        return v

def grad_N(a, K, x, y):
    '''Градиент узловой базисной функции в треугольнике с номером K, в узле с локальным номером a
    в точке (x, y). Вычисляется с помощью перехода к опорному элементу
    с узлами (0, 0), (0, 1), (1, 0).
    '''
    # градиенты базисных угловых функций в опорном элементе
    # определяем координаты треугольника
    x1, y1, x2, y2, x3, y3 = nodes_coords[triangles[K]].flatten()
    # матрица преобразования
    B = (np.array([[y3 - y1, -(y2 - y1)], [-(x3 - x1), x2 - x1]])
        / ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)))
    # возвращаем результат в зависимости от номера базисной функции
    if a == 0:
        return B @ np.array([-1, -1])
    elif a == 1:
        return B @ np.array([1, 0])
    elif a == 2:
        return B @ np.array([0, 1])


def triangle_mass_matrix(K):
    '''Вычисляет матрицу масс (см. отчёт) с максимум 9 ненулевыми элементами
    для треугольника с номером K по квадратурным формулам.
    '''
    # глобальные номера и координаты этих узлов треугольника
    tri_nums = triangles[K]
    tri_coords = nodes_coords[tri_nums]
    middles = np.array([nodes_coords[0] + nodes_coords[1],
                        nodes_coords[0] + nodes_coords[2],
                        nodes_coords[1] + nodes_coords[2]]) / 2
    # площадь треугольника
    x1, y1, x2, y2, x3, y3 = nodes_coords[triangles[K]].flatten()
    area = np.abs(((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))) / 2
    # инициализируем матрицу нулями
    T_K = np.zeros((nNod, nNod))
    for i in range(3):
        for j in range(3):
            T_K[tri_nums[i]][tri_nums[j]] += (area / 3
                    * (N(i, K, *middles[0]) * N(j, K, *middles[0])
                     + N(i, K, *middles[1]) * N(j, K, *middles[1])
                     + N(i, K, *middles[2]) * N(j, K, *middles[2])))
    return T_K


def triangle_stiff_matrix(K):
    '''Вычисляет матрицу жёсткости (см. отчёт) с максимум 9 ненулевыми элементами
    для треугольника с номером K по квадратурным формулам.
    '''
    # глобальные номера и координаты этих узлов треугольника
    tri_nums = triangles[K]
    tri_coords = nodes_coords[tri_nums]
    # площадь треугольника
    x1, y1, x2, y2, x3, y3 = nodes_coords[triangles[K]].flatten()
    area = np.abs(((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))) / 2
    # инициализируем матрицу нулями
    T_K = np.zeros((nNod, nNod))
    for i in range(3):
        for j in range(3):
            T_K[tri_nums[i]][tri_nums[j]] = area * grad_N(i, K, *tri_coords[i]) @ grad_N(j, K, *tri_coords[j])
    return T_K


def generate_W():
    '''Основная матрица жёсткости (см. отчёт).
    '''
    return np.sum(triangle_stiff_matrix(K) for K in range(len(triangles)))


def generate_M():
    '''Основная матрица масс (см. отчёт).
    '''
    return np.sum(triangle_mass_matrix(K) for K in range(len(triangles)))

def g(t):
    '''Возвращает вектор значений в узлах Дирихле на шаге времени t.
    '''
    return np.zeros(len(dir_nodes))


def f(t):
    '''Возвращает вектор значений f на шаге времени t.
    '''
    return np.zeros(len(ind_nodes))


def u0():
    '''Начальные данные.
    '''
    u0 = []
    for i in range(grid_density):
        for j in range(grid_density):
            x = i / (grid_density - 1)
            y = j / (grid_density - 1)
            u0.append(x * (1 - x) * y * (1 - y))
    return np.array(u0)


def step(u):
    '''Получает следующий вектор значений из предыдущего u.
    '''
    # левая часть
    A = T[:, ind_nodes][ind_nodes, :]
    # правая часть
    b = (M[ind_nodes, :] @ u
        - (W[:, dir_nodes][ind_nodes, :] * th + M[:, dir_nodes][ind_nodes, :]) @ g(t+1)
        - f(t))
    u_next_ind = np.linalg.solve(A, b)
    u_next = np.zeros(nNod)
    u_next[ind_nodes] = u_next_ind
    u_next[dir_nodes] = g(t+1)
    return u_next


def analytical_u(x, y, t):
    def _lambda(n, m):
        return - (np.pi ** 2) * (n ** 2 + m ** 2)
    def _sum(prec):
        res = 0
        for n in range(1, prec):
            for m in range(1, prec):
                res += (((-1) ** n - 1) * ((-1) ** m - 1)
                        * np.e ** (_lambda(n, m) * t)
                        * np.sin(np.pi * x * n) * np.sin(np.pi * y * m)
                        / n ** 3 / m ** 3)
        return res
    return 16 / (np.pi ** 6) * _sum(10)


# ==================================================================================================
# строим график скорости сходимости

diffs = []
for timesteps in range(4, 60):
    grid_density = 5
    th = t_max / timesteps
    # величина шага по пространству
    xh = (x_max - x_min) / grid_density
    yh = (y_max - y_min) / grid_density

    # общее количество узлов
    nNod = grid_density ** 2

    # dir_nodes - номера узлов с условием Дирихле, в данном случае внешние
    # ind_nodes - номера свободных, т.е. в данном случае внутренних узлов
    nodes_coords = generate_nodes(x_min, x_max, y_min, y_max, grid_density)
    dir_nodes = np.array([i + grid_density * j for j in range(grid_density) for i in range(grid_density)
                                                                    if i == 0 or i == grid_density - 1
                                                                    or j == 0 or j == grid_density - 1])
    ind_nodes = np.array([x for x in range(grid_density ** 2) if x not in dir_nodes])

    # триангулируем
    triangles = triangulate_rectangle(nodes_coords)

    # строим основные матрицу
    M = generate_M()
    W = generate_W()
    T = M + W * th
    T_dir = T[:, dir_nodes]
    T_ind = T[:, ind_nodes]

    # задаём начальные условия
    t = 0
    u = u0()

    # проходим по времени
    for i in range(timesteps):
        t += th
        u = step(u)

    # вычисляем аналитическое решение в узлах сетки
    analytical_us = []
    for i in range(grid_density):
        for j in range(grid_density):
            x = i / (grid_density - 1)
            y = j / (grid_density - 1)
            analytical_us.append(analytical_u(x, y, t))

    diffs.append(max(np.abs(analytical_us - u)) / th)

plt.plot(diffs[5:])
plt.show()
