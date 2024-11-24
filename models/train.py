from tqdm import tqdm
from models.fusion import FusionModel

import torch
from torch import nn

class FusionModelTrainer:

    def __init__(self, train_data, train_labels, test_data, test_labels, config={}, debug=False):
        self.debug = debug
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.model = FusionModel(304, 10)
        self.config = config
        self.epochs_trained = 0
        assert len(train_data) == len(train_labels), "Data and labels must have the same length."

    def _print_debug(self, message):
        if self.debug:
            print(message)

    def train(self):
        self._print_debug("Training Fusion Model")
        # Train the fusion model using the training data and labels
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get("learning_rate", 0.001))

        inputs = torch.tensor(self.train_data, dtype=torch.float32)
        labels = torch.tensor(self.train_labels, dtype=torch.long)

        num_epochs = self.config.get("num_epochs", 1000)
        for epoch in tqdm(range(num_epochs), desc="Training Fusion Model", initial=self.epochs_trained, total=self.epochs_trained + num_epochs):
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            self._print_debug(f"Epoch [{self.epochs_trained + epoch + 1}/{self.epochs_trained + num_epochs}], Loss: {loss.item():.4f}")
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