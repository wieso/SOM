import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets, preprocessing



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
            if i % 5000 == 0:
                print(i)
                self.heatmap()

    def heatmap(self):
        # TODO: форматирование графиков
        size = self.w.shape[1]
        nrows = int(np.sqrt(size)) + 1
        ncols = int(np.sqrt(size))
        f, ax = plt.subplots(nrows, ncols)
        for i, w in enumerate(self.w.transpose()):
            x = i // nrows
            y = i - i // ncols * ncols
            grid = np.reshape(w, (self.dim, self.dim))
            sns.heatmap(grid, ax=ax[y][x])
        neurons = np.sum(self.w, axis=1)
        grid = np.reshape(neurons, (self.dim, self.dim))
        plt.figure()
        sns.heatmap(grid)
        plt.show()

    def create_map(self, data, y):
        neurons = np.zeros((self.dim, self.dim))
        annot = np.chararray((self.dim, self.dim), unicode=True)
        annot[:] = ''
        for k, i in enumerate(data):
            r = self._win_neuron(i)
            x_coord = r // self.dim
            y_coord = r - r // self.dim * self.dim
            neurons[x_coord][y_coord] += 1
            annot[x_coord][y_coord] = y[k]

        sns.heatmap(neurons, annot=annot, fmt='')
        plt.show()

        weights = np.sum(self.w, axis=1)
        grid = np.reshape(weights, (self.dim, self.dim))
        sns.heatmap(grid, annot=annot, fmt='', cmap="hsv")
        plt.show()


iris = datasets.load_iris()
x = iris.data
y = iris.target

X_normalized = preprocessing.normalize(x, norm='l2')


dim = 25
s = SOM(4, dim)
s.training(X_normalized, 10000)


s.create_map(X_normalized, y=y)