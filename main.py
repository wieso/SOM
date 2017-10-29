import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets, preprocessing

iris = datasets.load_iris()
x = iris.data

# sigma = lambda sigma_0, n, teta: sigma_0 * np.math.exp(-n / teta)

X_normalized = preprocessing.normalize(x, norm='l2')


class SOM:
    def __init__(self, input_dim, dim=10, sigma=None, learning_rate=0.1, tay2=1000):
        self.dim = dim  # общее количество нейронов
        self.input_dim = input_dim  # размерность входного пространства
        self.sigma = sigma if sigma is not None else dim / 2
        self.learning_rate = learning_rate
        self.tay1 = 1000 / np.log(self.sigma)
        # минимальное значение сигма на шаге 1000 (определяем по формуле 3)
        self.minsigma = self.sigma * np.exp(-1000 / (1000 / np.log(self.sigma)))
        self.tay2 = tay2
        self.w = np.random.rand(dim * dim, input_dim)
        # матрица позиций всех нейронов, для определения латерального расстояния
        self.positions = np.argwhere(np.ones((dim, dim)) == 1)
        self.n = 1

    # Formula 1
    def _win_neuron(self, input_vector):
        distance = np.sqrt(np.sum(np.square(input_vector - self.w), axis=1))
        return np.argmin(distance)

    # Расстояния
    def _dist(self, win_index):
        d = np.sqrt(np.sum(
            np.square(self.positions - [win_index // self.dim, win_index - win_index // self.dim * self.dim]),
            axis=1))
        return d

    # Формула топологической окрестности
    def _topol(self, d):
        sigma = self.minsigma if self.n > 1000 else self.sigma * np.exp(-self.n / self.tay1)
        return np.exp(- np.square(d) / (2 * sigma * sigma))

    # Изменение весов
    def _change_weights(self, x, tnh):
        lr = self.learning_rate * np.exp(-self.n / self.tay2)
        minlr = 0.01
        lr = minlr if lr <= minlr else lr
        delta = np.transpose(lr * tnh * np.transpose(x - self.w))
        self.w = self.w + delta

    def _one_train(self, x):
        c = self._win_neuron(x)
        coop_dist = self._dist(c)
        t = self._topol(coop_dist)
        self._change_weights(x, t)
        self.n = self.n + 1

    def training(self, data, max_iteration=10000):
        for i in range(max_iteration):
            j = np.random.randint(0, len(data))
            x = data[j]
            self._one_train(x)
            if i % 10000 == 0:
                print(i)
                self.heatmap()

    def heatmap(self):
        neurons = np.sum(self.w, axis=1)
        grid = np.reshape(neurons, (self.dim, self.dim))
        sns.heatmap(grid)
        plt.show()


s = SOM(4, 50)
s.training(X_normalized, 20000)



# c = win_neuron(x_1, w)
# print(c)
# coop_dist = dist(c)
# t = topol(coop_dist, sigma(1))
# w = change_weights(x, w, 1, t)
# print(w)
