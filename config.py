import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

# координаты углов пластинки
x_min = 0
x_max = 1
y_min = 0
y_max = 1

# на сколько частей делится каждая сторона пластинки при построении сетки
grid_density = 5

# общее количество узлов
nNod = grid_density ** 2

timesteps = 10
