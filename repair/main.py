from maraboupy import Marabou, MarabouCore
import numpy as np
from pprint import pprint
import sys

network = Marabou.read_onnx("trained_model.onnx")

input_vars = network.inputVars[0].flatten()
output_vars = network.outputVars[0].flatten()

# Input bounds
for i in range(28):
    network.setLowerBound(input_vars[i], 0.0)
    network.setUpperBound(input_vars[i], 1.0)

network.setUpperBound(input_vars[22], 0.1)

# Property: output_2 >= all other outputs
output_2 = output_vars[1]
for j, out in enumerate(output_vars):
    if j != 1:
        network.addInequality([output_2, out], [-1, 1], -0.0001)

# eps = 1e-4
# output_2 = output_vars[1]

# disjuncts = []
# for j, out in enumerate(output_vars):
#     if j == 1:
#         continue
#     # Create inequality: output_j - output_2 <= -eps
#     eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
#     # eq.setType(MarabouCore.Equation.EquationType.LE)
#     eq.addAddend(1.0, out)
#     eq.addAddend(-1.0, output_2)
#     eq.setScalar(-eps)
#     disjuncts.append([eq])
#     print(eq.EquationType.name)
#     print(dir(eq))


# network.addDisjunctionConstraint(disjuncts)


options = Marabou.createOptions(verbosity=0)

counterexamples = []
MAX_CEX = 50
EPSILON = 1e-3

for k in range(MAX_CEX):
    print(f"\n🔍 Searching for counterexample #{k+1}...")
    res, vals, stats = network.solve(options=options)
    
    if not vals:
        print(f"✅ No more counterexamples found after {k} iterations.")
        break

    # Save and print this counterexample
    example = {v: vals[v] for v in input_vars}
    counterexamples.append(example)
    print(f"❌ Counterexample #{k+1} found:")
    for i, v in enumerate(input_vars):
        print(f"x{i+1} = {vals[v]:.4f}")

    # Add linear exclusion constraint:
    # Sum |x_i - val_i| >= EPSILON  ≈  sum_i (x_i - val_i) >= EPSILON  OR  sum_i (val_i - x_i) >= EPSILON
    # Marabou can’t express OR, so we alternate sign pattern to carve out different regions.
    coeffs = [1.0 for _ in input_vars]
    rhs = sum(vals[v] for v in input_vars)
    
    # Add constraint: sum(x_i) <= rhs - EPSILON
    network.addInequality(input_vars, [1.0]*len(input_vars), rhs - EPSILON)

print(f"\nTotal counterexamples found: {len(counterexamples)}")

pprint(counterexamples)

pprint("\n\n\n\n\n calculating gradients \n\n\n\n")

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnx2pytorch
from onnx2pytorch import ConvertModel

# ✅ Load your trained model from ONNX
onnx_model = onnx.load("trained_model.onnx")
model = ConvertModel(onnx_model)
model.eval()

# If GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Convert Marabou counterexamples (dicts) to tensor form
x_cexs = []
for ex in counterexamples:
    x = torch.tensor([ex[v] for v in sorted(ex.keys())], dtype=torch.float32)
    x_cexs.append(x.to(device))

# Function to get nearest satisfying output
def repair_output(y):
    y_rep = y.clone()
    y2 = y[1]
    y_rep[1] = -np.inf
    for j in range(len(y)):
        if j != 1:
            # y_rep[1] = y[j] + 1e-4  # enforce y2 >= yj + small margin
            y_rep[1] = max(y_rep[1], y[j] - 1e-4)
    return y_rep

# Collect layer-wise gradient norms
for idx, x_cex in enumerate(x_cexs):
    x_cex = x_cex.unsqueeze(0).requires_grad_(True)

    y = model(x_cex).squeeze(0)
    y_repaired = repair_output(y)

    print(y)
    print(y_repaired)

    # sys.exit()


    # L2 loss between original and repaired output
    loss = F.mse_loss(y, y_repaired)
    model.zero_grad()
    loss.backward()

    print(f"\n🧩 Counterexample #{idx+1}")
    print(f"L2 loss: {loss.item():.6f}")

    # Compute layer-wise gradient norms
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            print(f"Layer {name:25s} | grad norm = {grad_norm:.6f}")
