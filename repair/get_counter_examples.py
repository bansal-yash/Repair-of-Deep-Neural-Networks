from maraboupy import Marabou
from maraboupy import MarabouCore
import numpy as np
from pprint import pprint
import sys
# Load the ONNX model
network = Marabou.read_onnx("trained_model.onnx")

# Get input and output variables
inputVars = network.inputVars[0][0]  # Shape: [28]
outputVars = network.outputVars[0][0]    # Shape: [8]

print(f"Input variables: {len(inputVars)}")
print(f"Output variables: {len(outputVars)}")

# Set input constraints
# All inputs in range [0, 1]
for i in range(28):
    network.setLowerBound(inputVars[i], 0.0)
    network.setUpperBound(inputVars[i], 1.0)

# 22nd input (index 22) in range [0, 0.1]
network.setLowerBound(inputVars[22], 0.0)
network.setUpperBound(inputVars[22], 0.1)

# Property to verify: output[1] is NOT the smallest
disjuncts = []
for i in range(8):
    if i != 1:
        eq = Marabou.Equation()
        eq.addAddend(1.0, outputVars[1])
        eq.addAddend(-1.0, outputVars[i])
        eq.setScalar(0.0001)
        eq.EquationType = MarabouCore.Equation.GE
        disjuncts.append([eq])

network.addDisjunctionConstraint(disjuncts)

print("\nStarting verification...")
print("Searching for counterexamples where output[1] is NOT the smallest...")
print(f"Input constraints: All inputs in [0,1], input[22] in [0,0.1]\n")

# Store counterexamples
counterexamples = []
num_desired = 10
diversity_threshold = 0.15  # Minimum L2 distance between counterexamples

def calculate_distance(input1, input2):
    """Calculate L2 (Euclidean) distance between two input vectors"""
    return np.sqrt(sum((input1[i] - input2[i])**2 for i in range(len(input1))))

def is_diverse(new_input, existing_examples, threshold):
    """Check if new input is sufficiently different from existing ones"""
    for existing in existing_examples:
        if calculate_distance(new_input, existing) < threshold:
            return False
    return True

# Find multiple counterexamples
for attempt in range(num_desired * 10):  # Try up to 10x desired amount
    print(f"Attempt {attempt + 1}: ", end="", flush=True)
    
    # Solve the network
    exitCode, vals, stats = network.solve()
    
    if exitCode == "sat":
        # Extract input values
        input_vals = [vals[inputVars[i]] for i in range(28)]
        output_vals = [vals[outputVars[i]] for i in range(8)]
        
        # Check if this counterexample is diverse enough
        if is_diverse(input_vals, counterexamples, diversity_threshold):
            counterexamples.append(input_vals)
            print(f"✓ Found counterexample #{len(counterexamples)}")
            
            if len(counterexamples) >= num_desired:
                print(f"\n✓ Successfully found {num_desired} diverse counterexamples!")
                break
            
            # Add constraint to exclude neighborhood around this solution
            # Force at least one input to differ by at least diversity_threshold/sqrt(28)
            exclusion_disjuncts = []
            min_diff = diversity_threshold / np.sqrt(28)
            
            for i in range(28):
                # Either input[i] < current_value - min_diff
                eq1 = Marabou.Equation()
                eq1.addAddend(1.0, inputVars[i])
                eq1.setScalar(input_vals[i] - min_diff)
                eq1.EquationType = MarabouCore.Equation.LE
                
                # Or input[i] > current_value + min_diff
                eq2 = Marabou.Equation()
                eq2.addAddend(1.0, inputVars[i])
                eq2.setScalar(input_vals[i] + min_diff)
                eq2.EquationType = MarabouCore.Equation.GE
                
                exclusion_disjuncts.append([eq1])
                exclusion_disjuncts.append([eq2])
            
            network.addDisjunctionConstraint(exclusion_disjuncts)
        else:
            print("✗ Too similar to existing counterexample, adding exclusion constraint...")
            
            # Add stricter exclusion for this specific point
            exclusion_disjuncts = []
            for i in range(28):
                eq1 = Marabou.Equation()
                eq1.addAddend(1.0, inputVars[i])
                eq1.setScalar(input_vals[i] - 0.01)
                eq1.EquationType = MarabouCore.Equation.LE
                
                eq2 = Marabou.Equation()
                eq2.addAddend(1.0, inputVars[i])
                eq2.setScalar(input_vals[i] + 0.01)
                eq2.EquationType = MarabouCore.Equation.GE
                
                exclusion_disjuncts.append([eq1])
                exclusion_disjuncts.append([eq2])
            
            network.addDisjunctionConstraint(exclusion_disjuncts)
    
    elif exitCode == "unsat":
        print("✗ No more counterexamples found (UNSAT)")
        break
    else:
        print(f"✗ Unknown result: {exitCode}")
        break

# Display results
print(f"\n{'='*80}")
print(f"VERIFICATION COMPLETE")
print(f"{'='*80}")
print(f"Total counterexamples found: {len(counterexamples)}")

if len(counterexamples) == 0:
    print("\n✓ PROPERTY HOLDS!")
    print("Output[1] is always the smallest for the given input constraints.")
else:
    print("\n✗ PROPERTY VIOLATED!")
    print(f"Found {len(counterexamples)} diverse counterexamples where output[1] is NOT the smallest.\n")
    
    # Display all counterexamples
    for idx, input_vals in enumerate(counterexamples):
        # Reload and evaluate to get outputs
        test_network = Marabou.read_onnx("trained_model.onnx")
        for i in range(28):
            test_network.setLowerBound(test_network.inputVars[0][0][i], input_vals[i])
            test_network.setUpperBound(test_network.inputVars[0][0][i], input_vals[i])
        
        _, test_vals, _ = test_network.solve()
        output_vals = [test_vals[test_network.outputVars[0][0][i]] for i in range(8)]
        
        print(f"\n--- Counterexample {idx + 1} ---")
        print(f"Inputs (showing first 10):")
        for i in range(min(10, 28)):
            print(f"  input[{i}] = {input_vals[i]:.6f}")
        if len(input_vals) > 10:
            print(f"  ... ({len(input_vals) - 10} more inputs)")
        
        print(f"\nOutputs:")
        for i in range(8):
            marker = " ← output[1]" if i == 1 else ""
            print(f"  output[{i}] = {output_vals[i]:.6f}{marker}")
        
        min_idx = min(range(8), key=lambda i: output_vals[i])
        print(f"\nSmallest output: output[{min_idx}] = {output_vals[min_idx]:.6f}")
    
    # Compute diversity statistics
    if len(counterexamples) > 1:
        distances = []
        for i in range(len(counterexamples)):
            for j in range(i + 1, len(counterexamples)):
                distances.append(calculate_distance(counterexamples[i], counterexamples[j]))
        
        print(f"\n{'='*80}")
        print("DIVERSITY STATISTICS")
        print(f"{'='*80}")
        print(f"Average L2 distance between counterexamples: {np.mean(distances):.4f}")
        print(f"Minimum L2 distance: {np.min(distances):.4f}")
        print(f"Maximum L2 distance: {np.max(distances):.4f}")

pprint(counterexamples)

print("\n\n\n\n\n calculating gradients \n\n\n\n")

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
    # print(ex)
    x = torch.tensor(ex)
    x_cexs.append(x.to(device))

# Function to get nearest satisfying output
def repair_output(y):
    y_rep = y.clone()
    y2 = y[1]
    # y_rep[1] = -np.inf
    for j in range(len(y)):
        if j != 1:
            # y_rep[1] = y[j] + 1e-4  # enforce y2 >= yj + small margin
            y_rep[1] = min(y_rep[1], y[j] - 1e-4)
    return y_rep

# Collect layer-wise gradient norms
grad_sums = {}
grad_counts = {}

for idx, x_cex in enumerate(x_cexs):
    x_cex = x_cex.unsqueeze(0).requires_grad_(True)

    y = model(x_cex).squeeze(0)
    y_repaired = repair_output(y)

    # L2 loss between original and repaired output
    loss = F.mse_loss(y, y_repaired)
    model.zero_grad()
    loss.backward()

    print(f"\n🧩 Counterexample #{idx+1}")
    print(f"L2 loss: {loss.item():.6f}")

    # Collect gradient norms per layer
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            grad_sums[name] = grad_sums.get(name, 0.0) + grad_norm
            grad_counts[name] = grad_counts.get(name, 0) + 1
            print(f"Layer {name:25s} | grad norm = {grad_norm:.6f}")

# After all counterexamples, compute and print averages
print("\n📊 Average gradient norms across all counterexamples:")
for name in grad_sums:
    avg_norm = grad_sums[name] / grad_counts[name]
    print(f"Layer {name:25s} | avg grad norm = {avg_norm:.6f}")

print(counterexamples)
import csv

with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(counterexamples)