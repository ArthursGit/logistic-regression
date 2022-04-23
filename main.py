import numpy as np
import sklearn.datasets as ds
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.nn import Sigmoid

from linear_regression import PyTorchLinearModel, train_pytorch_model
from torch_helpers.TorchDataset import TorchDataset


def sigmoid(x):
    return 1 / (1 + torch.exp(-1 * x))


def cost_difference(prediction, y):
    return -1 * y * torch.log(prediction) - (1 - y) * torch.log(1 - prediction)


class LogisticRegressionModel:

    def __init__(self, n_features, n_classes=1):
        self.theta = torch.randn(n_features, n_classes)

    def forward(self, x):
        sigmoid_function = Sigmoid()
        return sigmoid_function(x @ self.theta)

    def evaluate(self, validation_data):
        total_cost = 0
        for x, y in validation_data:
            prediction = self.forward(x)
            total_cost += float(cost_difference(prediction, y).sum())
        total_cost /= len(validation_data)
        return total_cost

    def train(self, train_loader, n_epochs, learning_rate, validation_loader):
        costs = []
        for epoch in tqdm(range(n_epochs)):
            delta_theta = 0
            for x, y in train_loader:
                delta_theta += x.T @ (self.theta - y)
            self.theta -= learning_rate * delta_theta
            self.theta /= len(train_loader)
            costs.append(self.evaluate(validation_loader))
        return costs


def main():
    X, y = ds.make_classification(n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1,
                                  random_state=42)
    dataset = TorchDataset(X, y)

    validation_ration = 0.2
    batch_size = 32
    n_epochs = 5000
    learning_rate = 0.15

    train_loader, validation_loader = dataset.get_train_and_validation_data(validation_ration, batch_size)

    logistic_regression = LogisticRegressionModel(1)
    costs = logistic_regression.train(train_loader, n_epochs, learning_rate, validation_loader)
    plt.plot(costs)
    plt.show()

    plt.scatter(X, y)

    x_values = np.linspace(-3, 3)
    y_values = []
    for x_value in x_values:
        temp_tensor = torch.Tensor([x_value])
        y_values.append(float(logistic_regression.forward(temp_tensor)))

    plt.plot(x_values, y_values, color="red")
    plt.show()


if __name__ == "__main__":
    main()