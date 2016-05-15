import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# координаты углов пластинки
x_min = 0
x_max = 1
y_min = 0
y_max = 1

# шаг th по времени постоянен
# отсчёт начинается с t0 = 0
t_max = 1
timesteps = 10
th = t_max / timesteps
