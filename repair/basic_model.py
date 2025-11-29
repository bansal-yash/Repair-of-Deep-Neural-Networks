import torch
import torch.nn as nn

# ===== Define the model =====
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc2 = nn.Linear(2, 2, bias=False)
        self.relu = nn.ReLU()

        # Set weights as per the diagram
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor([[1.0, -2.0],
                                                [2.0, -1.0]]))
            self.fc2.weight.copy_(torch.tensor([[1.0, -1.0],
                                                [-1.0, 1.0]]))

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ===== Instantiate model =====
model = SimpleNet()
model.eval()

# ===== Dummy input for export =====
dummy_input = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

# ===== Export to ONNX =====
torch.onnx.export(
    model,
    dummy_input,
    "custom_network.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,
    opset_version=12
)

print("✅ Model saved as 'custom_network.onnx'")
