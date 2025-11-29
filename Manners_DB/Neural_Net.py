import torch
import torch.nn as nn
import numpy as np


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers):
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)

        hidden_layers = []
        for _ in range(num_hidden_layers - 1):
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        hidden_layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)

        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


def train_model(
    model: NeuralNet, train_loader, val_loader, criterion, optimizer, num_epochs
):
    patience = 20
    best_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model)
    print(f"Final Train Loss: {train_loss:.6f}")
    print(f"Final Validation Loss: {best_loss:.6f}")
