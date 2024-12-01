from tqdm import tqdm
from models.fusion import FusionModel

import torch
from torch import nn
import matplotlib.pyplot as plt

EVAL_RATE = 10
DEFAULT_CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 1000,
    "dropout_prob": 0.5
}

class FusionModelTrainer:

    def __init__(self, train_data, train_labels, test_data, test_labels, hidden_sizes=[], num_classes=10, config=DEFAULT_CONFIG, debug=False):
        self.debug = debug
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.model = FusionModel(input_size=train_data.shape[1], hidden_sizes=hidden_sizes, output_size=num_classes, dropout_prob=config.get("dropout_prob", DEFAULT_CONFIG["dropout_prob"]))
        self.config = config
        self.epochs_trained = 0
        assert len(train_data) == len(train_labels), "Data and labels must have the same length."

        self.train_accuracies = []
        self.test_accuracies = []

    def _print_debug(self, message):
        if self.debug:
            print(message)

    def train(self):
        self._print_debug("Training Fusion Model")
        # Train the fusion model using the training data and labels
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get("learning_rate", DEFAULT_CONFIG["learning_rate"]))

        inputs = torch.tensor(self.train_data, dtype=torch.float32)
        labels = torch.tensor(self.train_labels, dtype=torch.long)

        num_epochs = self.config.get("num_epochs", DEFAULT_CONFIG["num_epochs"])

        for epoch in tqdm(range(num_epochs), desc="Training Fusion Model", initial=self.epochs_trained, total=self.epochs_trained + num_epochs):
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            self._print_debug(f"Epoch [{self.epochs_trained + epoch + 1}/{self.epochs_trained + num_epochs}], Loss: {loss.item():.4f}")

            if epoch % EVAL_RATE == 0:
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == labels).sum().item() / len(labels)
                self.train_accuracies.append(accuracy)

                test_inputs = torch.tensor(self.test_data, dtype=torch.float32)
                test_labels = torch.tensor(self.test_labels, dtype=torch.long)
                test_outputs = self.model(test_inputs)
                _, test_predicted = torch.max(test_outputs, 1)
                test_accuracy = (test_predicted == test_labels).sum().item() / len(test_labels)
                self.test_accuracies.append(test_accuracy)

        self.epochs_trained += num_epochs

        self._print_debug("Model training completed")

    def evaluate(self):
        self._print_debug("Evaluating Fusion Model")
        with torch.no_grad():
            train_inputs = torch.tensor(self.train_data, dtype=torch.float32)
            train_labels = torch.tensor(self.train_labels, dtype=torch.long)
            test_inputs = torch.tensor(self.test_data, dtype=torch.float32)
            test_labels = torch.tensor(self.test_labels, dtype=torch.long)

            train_outputs = self.model(train_inputs)
            _, train_predicted = torch.max(train_outputs, 1)
            train_accuracy = (train_predicted == train_labels).sum().item() / len(train_labels)

            test_outputs = self.model(test_inputs)
            _, test_predicted = torch.max(test_outputs, 1)
            test_accuracy = (test_predicted == test_labels).sum().item() / len(test_labels)

            self._print_debug(f"Train Accuracy: {train_accuracy * 100:.2f}%")
            self._print_debug(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        return train_accuracy, test_accuracy

    def latest_accuracies(self):
        return self.train_accuracies[-1], self.test_accuracies[-1]

    def plot_accuracies(self):
        x = [i * EVAL_RATE for i in range(1, len(self.train_accuracies) + 1)]
        plt.plot(x, self.train_accuracies, label="Train")
        plt.plot(x, self.test_accuracies, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Fusion Model Training")
        plt.legend()
        plt.show()

    def save_plot(self, path):
        x = [i * EVAL_RATE for i in range(1, len(self.train_accuracies) + 1)]
        plt.plot(x, self.train_accuracies, label="Train")
        plt.plot(x, self.test_accuracies, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Fusion Model Training")
        plt.legend()
        plt.savefig(path)

    def get_test_inferences(self):
        self._print_debug("Getting test inferences")
        with torch.no_grad():
            test_inputs = torch.tensor(self.test_data, dtype=torch.float32)
            test_outputs = self.model(test_inputs)
            return test_outputs.numpy()