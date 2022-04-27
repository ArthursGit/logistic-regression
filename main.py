import numpy as np
import sklearn.datasets as ds
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.nn import Sigmoid, Module

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


class LogisticRegressionPytorchModel(Module):

    def __init__(self, n_features, n_output):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, n_output)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def main():
    X, y = ds.make_classification(n_features=1, n_samples=500, n_informative=1, n_redundant=0, n_clusters_per_class=1,
                                  random_state=42)
    dataset = TorchDataset(X, y)

    validation_ration = 0.2
    batch_size = 64
    n_epochs = 1000
    learning_rate = 0.1

    n_features = 1
    n_output = 1
    train_loader, validation_loader = dataset.get_train_and_validation_data(validation_ration, batch_size)

    logistic_regression = LogisticRegressionModel(n_features)
    costs = logistic_regression.train(train_loader, n_epochs, learning_rate, validation_loader)

    pytorch_model = LogisticRegressionPytorchModel(n_features, n_output)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(pytorch_model.parameters(), learning_rate)

    # Training loop

    for epoch in tqdm(range(n_epochs)):
        for x_batch, y_batch in train_loader:
            prediction = pytorch_model(x_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    # Validation

    plt.plot(costs, color="green")

    x_values = np.linspace(-3, 3)
    y_values = []
    for x_value in x_values:
        temp_tensor = torch.Tensor([x_value])
        y_values.append(float(logistic_regression.forward(temp_tensor)))

    costs = []
    with torch.no_grad():
        for x_batch, y_batch in validation_loader:
            prediction = pytorch_model(x_batch)
            loss = criterion(prediction, y_batch)
            costs.append(loss.item())
    plt.show()

    plt.plot(costs, color="red")
    plt.show()

    plt.scatter(X, y)
    plt.plot(x_values, y_values, color="red")
    x_values = np.linspace(-3, 3)
    y_values = []
    with torch.no_grad():
        for x_batch in x_values:
            prediction = pytorch_model(torch.tensor([x_batch], dtype=torch.float32))
            y_values += [i.item() for i in prediction]

    plt.plot(x_values, y_values, color="green")
    plt.show()


if __name__ == "__main__":
    main()