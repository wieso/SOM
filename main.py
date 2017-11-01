import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets, preprocessing

from tqdm import tqdm


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

    def training(self, data, max_iteration=10000, show_stage=True, stage_step=5000):
        for i in tqdm(range(max_iteration)):
            j = np.random.randint(0, len(data))
            x = data[j]
            self._one_train(x)
            if show_stage and i % stage_step == 0:
                self.heatmap()

    def heatmap(self):
        # TODO: форматирование графиков
        size = self.w.shape[1]
        if np.sqrt(size) % 1 == 0:
            nrows = int(np.sqrt(size))
        else:
            nrows = int(np.sqrt(size)) + 1
        ncols = int(np.sqrt(size))
        f, axs = plt.subplots(nrows, ncols)
        axs = np.array(axs)
        for ax, w in zip(axs.reshape(-1), self.w.transpose()):
            grid = np.reshape(w, (self.dim, self.dim))
            sns.heatmap(grid, ax=ax)
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


def dataset_processing(dataset, dim: int = 25, epoch: int = 10000, **kwargs):
    x, y = dataset.data, dataset.target
    x_normalized = preprocessing.normalize(x, norm='l2')
    s = SOM(len(x_normalized[0]), dim)
    s.training(x_normalized, max_iteration=epoch, **kwargs)
    s.create_map(x_normalized, y=y)


if __name__ == '__main__':
    dataset = datasets.load_breast_cancer()
    dataset_processing(dataset, dim=50, epoch=100000, show_stage=False)
