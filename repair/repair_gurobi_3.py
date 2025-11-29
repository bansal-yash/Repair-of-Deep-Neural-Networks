import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sys
import torch
import torch.nn as nn


def create_neural_network():
    # Create a new model
    model = gp.Model("neural_network")
    # Suppress output (set to 1 to see solver output)
    model.setParam("OutputFlag", 0)

    # Define weight variables (delta weights to be added to base weights)
    # Layer 1->2: 2 inputs × 2 neurons = 4 weights
    # Layer 2->3: 2 neurons × 2 neurons = 4 weights
    # Layer 3->4: 2 neurons × 2 neurons = 4 weights
    # Layer 4->5: 2 neurons × 2 outputs = 4 weights
    # Total: 16 weights
    w = {}
    for i in range(1, 17):
        w[i] = model.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"w{i}"
        )

    # Layer 1 (input layer) - 2 inputs
    v1_1 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v1_1"
    )
    v1_2 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v1_2"
    )

    # Layer 2 - pre-activation
    z2_1 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z2_1"
    )
    z2_2 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z2_2"
    )

    # Layer 2 - after ReLU activation
    v2_1 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v2_1")
    v2_2 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v2_2")

    # Layer 3 - pre-activation
    z3_1 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z3_1"
    )
    z3_2 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z3_2"
    )

    # Layer 3 - after ReLU activation
    v3_1 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v3_1")
    v3_2 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v3_2")

    # Layer 4 - pre-activation
    z4_1 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z4_1"
    )
    z4_2 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z4_2"
    )

    # Layer 4 - after ReLU activation
    v4_1 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v4_1")
    v4_2 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v4_2")

    # Layer 5 (output layer) - no activation
    v5_1 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v5_1"
    )
    v5_2 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v5_2"
    )

    # Update model to integrate new variables
    model.update()

    # Base weights (randomly chosen)
    # Layer 1 to Layer 2 connections (fully connected: 2×2 = 4 weights)
    # Base weights: [[0.5, -0.3], [0.8, 0.2]]
    model.addConstr(
        z2_1 == (0.5 + w[1]) * v1_1 + (-0.3 + w[2]) * v1_2, name="z2_1_calc"
    )
    model.addConstr(z2_2 == (0.8 + w[3]) * v1_1 + (0.2 + w[4]) * v1_2, name="z2_2_calc")
    model.addGenConstrMax(v2_1, [z2_1, 0.0], name="relu_2_1")
    model.addGenConstrMax(v2_2, [z2_2, 0.0], name="relu_2_2")

    # Layer 2 to Layer 3 connections (fully connected: 2×2 = 4 weights)
    # Base weights: [[1.2, -0.7], [0.4, 1.5]]
    model.addConstr(
        z3_1 == (1.2 + w[5]) * v2_1 + (-0.7 + w[6]) * v2_2, name="z3_1_calc"
    )
    model.addConstr(z3_2 == (0.4 + w[7]) * v2_1 + (1.5 + w[8]) * v2_2, name="z3_2_calc")
    model.addGenConstrMax(v3_1, [z3_1, 0.0], name="relu_3_1")
    model.addGenConstrMax(v3_2, [z3_2, 0.0], name="relu_3_2")

    # Layer 3 to Layer 4 connections (fully connected: 2×2 = 4 weights)
    # Base weights: [[-0.5, 0.9], [1.1, -0.4]]
    model.addConstr(
        z4_1 == (-0.5 + w[9]) * v3_1 + (0.9 + w[10]) * v3_2, name="z4_1_calc"
    )
    model.addConstr(
        z4_2 == (1.1 + w[11]) * v3_1 + (-0.4 + w[12]) * v3_2, name="z4_2_calc"
    )
    model.addGenConstrMax(v4_1, [z4_1, 0.0], name="relu_4_1")
    model.addGenConstrMax(v4_2, [z4_2, 0.0], name="relu_4_2")

    # Layer 4 to Layer 5 connections (fully connected: 2×2 = 4 weights)
    # Base weights: [[0.6, -1.0], [-0.8, 0.7]]
    model.addConstr(
        v5_1 == (0.6 + w[13]) * v4_1 + (-1.0 + w[14]) * v4_2, name="v5_1_calc"
    )
    model.addConstr(
        v5_2 == (-0.8 + w[15]) * v4_1 + (0.7 + w[16]) * v4_2, name="v5_2_calc"
    )

    return model, {
        "weights": w,
        "layer1": (v1_1, v1_2),
        "layer2": (v2_1, v2_2),
        "layer3": (v3_1, v3_2),
        "layer4": (v4_1, v4_2),
        "layer5": (v5_1, v5_2),
        "layer2_pre_activation": (z2_1, z2_2),
        "layer3_pre_activation": (z3_1, z3_2),
        "layer4_pre_activation": (z4_1, z4_2),
    }


def find_optimal_weights(model, variables):
    print(variables)

    # Fix input values
    v1_1, v1_2 = variables["layer1"]
    model.addConstr(v1_1 == 1, name="input_1_fixed_opt")
    model.addConstr(v1_2 == 2, name="input_2_fixed_opt")

    # Add constraint that output_2 > output_1
    v5_1, v5_2 = variables["layer5"]
    # model.addConstr(v5_2 >= v5_1 - 1.5, name="output_2_greater_than_output_1_by_1.5")

    # Create auxiliary variable for the maximum absolute value of all weights
    max_abs = model.addVar(
        lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="max_abs_weight"
    )

    # Add constraints: max_abs >= |wi| for all weights (1 to 16)
    for i in range(1, 17):
        model.addConstr(max_abs >= variables["weights"][i], name=f"max_abs_pos_{i}")
        model.addConstr(max_abs >= -variables["weights"][i], name=f"max_abs_neg_{i}")

    # # Create auxiliary variables for absolute values of all weights
    # abs_weights = {}
    # for i in range(1, 17):
    #     abs_weights[i] = model.addVar(
    #         lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"abs_w{i}"
    #     )
    #     # Add constraints: abs_weights[i] >= |w[i]|
    #     model.addConstr(abs_weights[i] >= variables["weights"][i], name=f"abs_pos_{i}")
    #     model.addConstr(abs_weights[i] >= -variables["weights"][i], name=f"abs_neg_{i}")

    # Optional: Fix certain weights to zero if needed
    for i in range(1, 17):
        model.addConstr(variables["weights"][i] == 0, name=f"fix_weights_{i}")

    # Set objective: minimize the maximum absolute value
    model.setObjective(max_abs, GRB.MINIMIZE)

    # total_change = sum(abs_weights[i] for i in range(1, 17))
    # model.setObjective(total_change, GRB.MINIMIZE)

    # Optimize
    model.setParam("OutputFlag", 1)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("\n" + "=" * 60)
        print("OPTIMAL SOLUTION FOUND")
        print("=" * 60)
        print(f"\nInput values:")
        print(f"  v1_1 = {v1_1.X:.4f}")
        print(f"  v1_2 = {v1_2.X:.4f}")

        print(f"\nAll weights:")
        total_sum = 0
        for i in range(1, 17):
            print(f"  w{i} = {variables['weights'][i].X:.6f}")
            total_sum += abs(variables["weights"][i].X)

        # print(f"\nTotal sum of absolute weight changes: {total_sum:.6f}")
        print(f"\nMaximum absolute weight value: {max_abs.X:.6f}")
        # Get pre-activation variables
        z2_1, z2_2 = variables["layer2_pre_activation"]
        z3_1, z3_2 = variables["layer3_pre_activation"]
        z4_1, z4_2 = variables["layer4_pre_activation"]

        print(f"\nLayer 2 pre-activations (before ReLU):")
        print(f"  z2_1 = {z2_1.X:.6f}")
        print(f"  z2_2 = {z2_2.X:.6f}")

        print(f"\nLayer 2 post-activations (after ReLU):")
        print(f"  v2_1 = {variables['layer2'][0].X:.6f}")
        print(f"  v2_2 = {variables['layer2'][1].X:.6f}")

        print(f"\nLayer 3 pre-activations (before ReLU):")
        print(f"  z3_1 = {z3_1.X:.6f}")
        print(f"  z3_2 = {z3_2.X:.6f}")

        print(f"\nLayer 3 post-activations (after ReLU):")
        print(f"  v3_1 = {variables['layer3'][0].X:.6f}")
        print(f"  v3_2 = {variables['layer3'][1].X:.6f}")

        print(f"\nLayer 4 pre-activations (before ReLU):")
        print(f"  z4_1 = {z4_1.X:.6f}")
        print(f"  z4_2 = {z4_2.X:.6f}")

        print(f"\nLayer 4 post-activations (after ReLU):")
        print(f"  v4_1 = {variables['layer4'][0].X:.6f}")
        print(f"  v4_2 = {variables['layer4'][1].X:.6f}")

        print(f"\nOutput layer (Layer 5 - no activation):")
        print(f"  v5_1 (output_1) = {v5_1.X:.6f}")
        print(f"  v5_2 (output_2) = {v5_2.X:.6f}")
        print(f"  Difference (v5_2 - v5_1) = {v5_2.X - v5_1.X:.6f}")
        print("=" * 60)
    else:
        print(f"Optimization failed with status: {model.status}")


def gradient_based_update(model, variables):
    # Fix input values
    v1_1, v1_2 = variables["layer1"]
    input_val_1 = 1.0
    input_val_2 = 2.0

    model.addConstr(v1_1 == input_val_1, name="input_1_grad_fixed")
    model.addConstr(v1_2 == input_val_2, name="input_2_grad_fixed")

    # Fix all weights to 0 (use base network)
    for i in range(1, 17):
        model.addConstr(variables["weights"][i] == 0, name=f"fix_weights_{i}")

    # Step 1: Calculate current output (forward pass with current weights)
    model.setObjective(0, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"Forward pass failed with status: {model.status}")
        return None

    # Get current output values
    current_output_1 = variables["layer5"][0].X
    current_output_2 = variables["layer5"][1].X

    # Get all pre-activations
    current_z2_1 = variables["layer2_pre_activation"][0].X
    current_z2_2 = variables["layer2_pre_activation"][1].X
    current_z3_1 = variables["layer3_pre_activation"][0].X
    current_z3_2 = variables["layer3_pre_activation"][1].X
    current_z4_1 = variables["layer4_pre_activation"][0].X
    current_z4_2 = variables["layer4_pre_activation"][1].X

    # Get all post-activations
    current_v2_1 = variables["layer2"][0].X
    current_v2_2 = variables["layer2"][1].X
    current_v3_1 = variables["layer3"][0].X
    current_v3_2 = variables["layer3"][1].X
    current_v4_1 = variables["layer4"][0].X
    current_v4_2 = variables["layer4"][1].X

    # Get current weights (all should be 0, but get them anyway)
    current_weights = {i: variables["weights"][i].X for i in range(1, 17)}

    print("\n" + "=" * 60)
    print("GRADIENT-BASED UPDATE")
    print("=" * 60)
    print(f"\nStep 1: Current Forward Pass")
    print(f"  Input: v1_1 = {input_val_1}, v1_2 = {input_val_2}")

    print(f"\n  Layer 2:")
    print(f"    Pre-activation: z2_1 = {current_z2_1:.6f}, z2_2 = {current_z2_2:.6f}")
    print(f"    Post-activation: v2_1 = {current_v2_1:.6f}, v2_2 = {current_v2_2:.6f}")

    print(f"\n  Layer 3:")
    print(f"    Pre-activation: z3_1 = {current_z3_1:.6f}, z3_2 = {current_z3_2:.6f}")
    print(f"    Post-activation: v3_1 = {current_v3_1:.6f}, v3_2 = {current_v3_2:.6f}")

    print(f"\n  Layer 4:")
    print(f"    Pre-activation: z4_1 = {current_z4_1:.6f}, z4_2 = {current_z4_2:.6f}")
    print(f"    Post-activation: v4_1 = {current_v4_1:.6f}, v4_2 = {current_v4_2:.6f}")

    print(f"\n  Output (Layer 5):")
    print(f"    v5_1 = {current_output_1:.6f}, v5_2 = {current_output_2:.6f}")
    print(f"    Difference (v5_2 - v5_1) = {current_output_2 - current_output_1:.6f}")
    print(
        f"  Constraint satisfied? v5_2 >= v5_1 + 1.5: {current_output_2 >= current_output_1 + 1.5}"
    )

    # Step 2: Find nearest output satisfying constraint (minimum MSE)
    # Create new variables for target output
    target_v5_1 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="target_v5_1"
    )
    target_v5_2 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="target_v5_2"
    )

    # Add constraint: target_v5_2 >= target_v5_1 + 1.5
    model.addConstr(target_v5_2 >= target_v5_1 - 1.5, name="target_constraint")

    # Minimize MSE
    diff_1 = target_v5_1 - current_output_1
    diff_2 = target_v5_2 - current_output_2

    model.setObjective(diff_1 * diff_1 + diff_2 * diff_2, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"Target optimization failed with status: {model.status}")
        return None

    target_output_1 = target_v5_1.X
    target_output_2 = target_v5_2.X
    mse = (target_output_1 - current_output_1) ** 2 + (
        target_output_2 - current_output_2
    ) ** 2

    print(f"\nStep 2: Nearest Output Satisfying Constraint")
    print(
        f"  Target Output: v5_1 = {target_output_1:.6f}, v5_2 = {target_output_2:.6f}"
    )
    print(f"  Difference (v5_2 - v5_1) = {target_output_2 - target_output_1:.6f}")
    print(f"  MSE Loss: {mse:.6f}")
    print(
        f"  Constraint satisfied? v5_2 >= v5_1 + 1.5: {target_output_2 >= target_output_1 + 1.5}"
    )

    # Step 3: Calculate gradients using PyTorch
    print(f"\nStep 3: Gradient Calculation (PyTorch)")

    # Layer 2 gradients
    z2_1_tensor = torch.tensor(current_z2_1, requires_grad=True)
    z2_2_tensor = torch.tensor(current_z2_2, requires_grad=True)
    v2_1_tensor = torch.relu(z2_1_tensor)
    v2_2_tensor = torch.relu(z2_2_tensor)

    # Forward pass through remaining layers
    # Layer 3
    w5, w6, w7, w8 = (
        current_weights[5],
        current_weights[6],
        current_weights[7],
        current_weights[8],
    )
    z3_1_tensor = (1.2 + w5) * v2_1_tensor + (-0.7 + w6) * v2_2_tensor
    z3_2_tensor = (0.4 + w7) * v2_1_tensor + (1.5 + w8) * v2_2_tensor
    v3_1_tensor = torch.relu(z3_1_tensor)
    v3_2_tensor = torch.relu(z3_2_tensor)

    # Layer 4
    w9, w10, w11, w12 = (
        current_weights[9],
        current_weights[10],
        current_weights[11],
        current_weights[12],
    )
    z4_1_tensor = (-0.5 + w9) * v3_1_tensor + (0.9 + w10) * v3_2_tensor
    z4_2_tensor = (1.1 + w11) * v3_1_tensor + (-0.4 + w12) * v3_2_tensor
    v4_1_tensor = torch.relu(z4_1_tensor)
    v4_2_tensor = torch.relu(z4_2_tensor)

    # Layer 5 (output)
    w13, w14, w15, w16 = (
        current_weights[13],
        current_weights[14],
        current_weights[15],
        current_weights[16],
    )
    v5_1_tensor = (0.6 + w13) * v4_1_tensor + (-1.0 + w14) * v4_2_tensor
    v5_2_tensor = (-0.8 + w15) * v4_1_tensor + (0.7 + w16) * v4_2_tensor

    # Calculate loss
    target_v5_1_tensor = torch.tensor(target_output_1)
    target_v5_2_tensor = torch.tensor(target_output_2)
    loss = (v5_1_tensor - target_v5_1_tensor) ** 2 + (
        v5_2_tensor - target_v5_2_tensor
    ) ** 2

    # Backpropagate
    loss.backward()
    grad_z2_1 = z2_1_tensor.grad.item()
    grad_z2_2 = z2_2_tensor.grad.item()

    print(f"\n  Layer 2 Gradients:")
    print(f"    ∂Loss/∂z2_1 = {grad_z2_1:.6f}")
    print(f"    ∂Loss/∂z2_2 = {grad_z2_2:.6f}")

    # Layer 3 gradients
    z3_1_tensor = torch.tensor(current_z3_1, requires_grad=True)
    z3_2_tensor = torch.tensor(current_z3_2, requires_grad=True)
    v3_1_tensor = torch.relu(z3_1_tensor)
    v3_2_tensor = torch.relu(z3_2_tensor)

    # Forward pass through remaining layers
    z4_1_tensor = (-0.5 + w9) * v3_1_tensor + (0.9 + w10) * v3_2_tensor
    z4_2_tensor = (1.1 + w11) * v3_1_tensor + (-0.4 + w12) * v3_2_tensor
    v4_1_tensor = torch.relu(z4_1_tensor)
    v4_2_tensor = torch.relu(z4_2_tensor)

    v5_1_tensor = (0.6 + w13) * v4_1_tensor + (-1.0 + w14) * v4_2_tensor
    v5_2_tensor = (-0.8 + w15) * v4_1_tensor + (0.7 + w16) * v4_2_tensor

    loss = (v5_1_tensor - target_v5_1_tensor) ** 2 + (
        v5_2_tensor - target_v5_2_tensor
    ) ** 2
    loss.backward()
    grad_z3_1 = z3_1_tensor.grad.item()
    grad_z3_2 = z3_2_tensor.grad.item()

    print(f"\n  Layer 3 Gradients:")
    print(f"    ∂Loss/∂z3_1 = {grad_z3_1:.6f}")
    print(f"    ∂Loss/∂z3_2 = {grad_z3_2:.6f}")

    # Layer 4 gradients
    z4_1_tensor = torch.tensor(current_z4_1, requires_grad=True)
    z4_2_tensor = torch.tensor(current_z4_2, requires_grad=True)
    v4_1_tensor = torch.relu(z4_1_tensor)
    v4_2_tensor = torch.relu(z4_2_tensor)

    v5_1_tensor = (0.6 + w13) * v4_1_tensor + (-1.0 + w14) * v4_2_tensor
    v5_2_tensor = (-0.8 + w15) * v4_1_tensor + (0.7 + w16) * v4_2_tensor

    loss = (v5_1_tensor - target_v5_1_tensor) ** 2 + (
        v5_2_tensor - target_v5_2_tensor
    ) ** 2
    loss.backward()
    grad_z4_1 = z4_1_tensor.grad.item()
    grad_z4_2 = z4_2_tensor.grad.item()

    print(f"\n  Layer 4 Gradients:")
    print(f"    ∂Loss/∂z4_1 = {grad_z4_1:.6f}")
    print(f"    ∂Loss/∂z4_2 = {grad_z4_2:.6f}")

    print("=" * 60)

    return {
        "current_output": (current_output_1, current_output_2),
        "target_output": (target_output_1, target_output_2),
        "current_z2": (current_z2_1, current_z2_2),
        "current_z3": (current_z3_1, current_z3_2),
        "current_z4": (current_z4_1, current_z4_2),
        "current_v2": (current_v2_1, current_v2_2),
        "current_v3": (current_v3_1, current_v3_2),
        "current_v4": (current_v4_1, current_v4_2),
        "mse_loss": mse,
        "gradients": {
            "dLoss_dz2_1": grad_z2_1,
            "dLoss_dz2_2": grad_z2_2,
            "dLoss_dz3_1": grad_z3_1,
            "dLoss_dz3_2": grad_z3_2,
            "dLoss_dz4_1": grad_z4_1,
            "dLoss_dz4_2": grad_z4_2,
        },
        "current_weights": current_weights,
    }


def gradient_based_update_multi(model, variables):

    allow_layer1 = True
    allow_layer2 = False
    allow_layer3 = False
    allow_layer4 = True
    """
    Perform gradient-based update with perturbations at layer 3.
    Calculates gradients, perturbs z3_1 and z3_2, and applies single layer modifications using Gurobi.
    """
    # Fix input values
    v1_1, v1_2 = variables["layer1"]
    input_val_1 = 1.0
    input_val_2 = 2.0

    model.addConstr(v1_1 == input_val_1, name="input_1_grad_multi_fixed")
    model.addConstr(v1_2 == input_val_2, name="input_2_grad_multi_fixed")

    # Fix all weights to 0 (use base network)
    for i in range(1, 17):
        model.addConstr(variables["weights"][i] == 0, name=f"fix_weights_multi_{i}")

    # Step 1: Calculate current output (forward pass with current weights)
    model.setObjective(0, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"Forward pass failed with status: {model.status}")
        return None

    # Get current output values
    current_output_1 = variables["layer5"][0].X
    current_output_2 = variables["layer5"][1].X

    # Get all pre-activations
    current_z2_1 = variables["layer2_pre_activation"][0].X
    current_z2_2 = variables["layer2_pre_activation"][1].X
    current_z3_1 = variables["layer3_pre_activation"][0].X
    current_z3_2 = variables["layer3_pre_activation"][1].X
    current_z4_1 = variables["layer4_pre_activation"][0].X
    current_z4_2 = variables["layer4_pre_activation"][1].X

    # Get all post-activations
    current_v2_1 = variables["layer2"][0].X
    current_v2_2 = variables["layer2"][1].X
    current_v3_1 = variables["layer3"][0].X
    current_v3_2 = variables["layer3"][1].X
    current_v4_1 = variables["layer4"][0].X
    current_v4_2 = variables["layer4"][1].X

    # Get current weights (all should be 0)
    current_weights = {i: variables["weights"][i].X for i in range(1, 17)}

    print("\n" + "=" * 60)
    print("GRADIENT-BASED UPDATE MULTI (WITH LAYER 3 PERTURBATION)")
    print("=" * 60)
    print(f"\nStep 1: Current Forward Pass")
    print(f"  Input: v1_1 = {input_val_1}, v1_2 = {input_val_2}")
    print(f"\n  Layer 2:")
    print(f"    Pre-activation: z2_1 = {current_z2_1:.6f}, z2_2 = {current_z2_2:.6f}")
    print(f"    Post-activation: v2_1 = {current_v2_1:.6f}, v2_2 = {current_v2_2:.6f}")
    print(f"\n  Layer 3:")
    print(f"    Pre-activation: z3_1 = {current_z3_1:.6f}, z3_2 = {current_z3_2:.6f}")
    print(f"    Post-activation: v3_1 = {current_v3_1:.6f}, v3_2 = {current_v3_2:.6f}")
    print(f"\n  Layer 4:")
    print(f"    Pre-activation: z4_1 = {current_z4_1:.6f}, z4_2 = {current_z4_2:.6f}")
    print(f"    Post-activation: v4_1 = {current_v4_1:.6f}, v4_2 = {current_v4_2:.6f}")
    print(f"\n  Output (Layer 5):")
    print(f"    v5_1 = {current_output_1:.6f}, v5_2 = {current_output_2:.6f}")
    print(f"    Difference (v5_2 - v5_1) = {current_output_2 - current_output_1:.6f}")

    # Step 2: Find nearest output satisfying constraint
    target_v5_1 = model.addVar(
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="target_v5_1_multi",
    )
    target_v5_2 = model.addVar(
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="target_v5_2_multi",
    )

    model.addConstr(target_v5_2 >= target_v5_1 - 1.5, name="target_constraint_multi")

    diff_1 = target_v5_1 - current_output_1
    diff_2 = target_v5_2 - current_output_2

    model.setObjective(diff_1 * diff_1 + diff_2 * diff_2, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"Target optimization failed with status: {model.status}")
        return None

    target_output_1 = target_v5_1.X
    target_output_2 = target_v5_2.X
    mse = (target_output_1 - current_output_1) ** 2 + (
        target_output_2 - current_output_2
    ) ** 2

    print(f"\nStep 2: Nearest Output Satisfying Constraint")
    print(
        f"  Target Output: v5_1 = {target_output_1:.6f}, v5_2 = {target_output_2:.6f}"
    )
    print(f"  Difference (v5_2 - v5_1) = {target_output_2 - target_output_1:.6f}")
    print(f"  MSE Loss: {mse:.6f}")

    # Step 3: Calculate gradients using PyTorch - focusing on z3_1 and z3_2
    print(f"\nStep 3: Gradient Calculation (PyTorch) - Focus on Layer 3")

    # Create tensors for z3_1 and z3_2 with gradient tracking
    z3_1_tensor = torch.tensor(current_z3_1, requires_grad=True)
    z3_2_tensor = torch.tensor(current_z3_2, requires_grad=True)
    v3_1_tensor = torch.relu(z3_1_tensor)
    v3_2_tensor = torch.relu(z3_2_tensor)

    v3_1_tensor.retain_grad()
    v3_2_tensor.retain_grad()
    # Forward pass through remaining layers (Layer 4 and 5)
    w9, w10, w11, w12 = (
        current_weights[9],
        current_weights[10],
        current_weights[11],
        current_weights[12],
    )
    z4_1_tensor = (-0.5 + w9) * v3_1_tensor + (0.9 + w10) * v3_2_tensor
    z4_2_tensor = (1.1 + w11) * v3_1_tensor + (-0.4 + w12) * v3_2_tensor
    v4_1_tensor = torch.relu(z4_1_tensor)
    v4_2_tensor = torch.relu(z4_2_tensor)

    w13, w14, w15, w16 = (
        current_weights[13],
        current_weights[14],
        current_weights[15],
        current_weights[16],
    )
    v5_1_tensor = (0.6 + w13) * v4_1_tensor + (-1.0 + w14) * v4_2_tensor
    v5_2_tensor = (-0.8 + w15) * v4_1_tensor + (0.7 + w16) * v4_2_tensor

    # Calculate loss
    target_v5_1_tensor = torch.tensor(target_output_1)
    target_v5_2_tensor = torch.tensor(target_output_2)
    loss = (v5_1_tensor - target_v5_1_tensor) ** 2 + (
        v5_2_tensor - target_v5_2_tensor
    ) ** 2

    # Backpropagate to get gradients w.r.t. z3_1 and z3_2
    loss.backward()
    grad_z3_1 = z3_1_tensor.grad.item()
    grad_z3_2 = z3_2_tensor.grad.item()

    grad_v3_1 = v3_1_tensor.grad.item()
    grad_v3_2 = v3_2_tensor.grad.item()

    print(f"\n  Derivatives w.r.t. Layer 3 Pre-activations:")
    print(f"    ∂Loss/∂z3_1 = {grad_z3_1:.6f}")
    print(f"    ∂Loss/∂z3_2 = {grad_z3_2:.6f}")
    print(f"\n  Derivatives w.r.t. Layer 3 Post-activations (after ReLU):")
    print(f"    ∂Loss/∂v3_1 = {grad_v3_1:.6f}")
    print(f"    ∂Loss/∂v3_2 = {grad_v3_2:.6f}")

    # Step 4: Perturb z3_1 and z3_2
    alpha = 0.4  # Learning rate / perturbation factor
    perturbed_z3_1 = current_z3_1 - alpha * grad_z3_1
    perturbed_z3_2 = current_z3_2 - alpha * grad_z3_2

    # perturbed_z3_1 = current_z3_1 - alpha * grad_v3_1
    # perturbed_z3_2 = current_z3_2 - alpha * grad_v3_2

    print(f"\nStep 4: Perturbation of Layer 3 Pre-activations")
    print(f"  Alpha (learning rate): {alpha}")
    print(
        f"  Original z3_1 = {current_z3_1:.6f} → Perturbed z3_1 = {perturbed_z3_1:.6f}"
    )
    print(
        f"  Original z3_2 = {current_z3_2:.6f} → Perturbed z3_2 = {perturbed_z3_2:.6f}"
    )
    print(f"  Change in z3_1: {perturbed_z3_1 - current_z3_1:.6f}")
    print(f"  Change in z3_2: {perturbed_z3_2 - current_z3_2:.6f}")

    # Step 5a: Single Layer Modification - Before Layer 3 (Layers 1→2 and 2→3) using Gurobi
    print(f"\nStep 5a: Single Layer Modification - Before Layer 3 (Using Gurobi)")
    print(f"  Goal: Modify weights in layers before layer 3 to achieve perturbed z3 values")
    print(f"  Allowed modifications: Layer1→2: {allow_layer1}, Layer2→3: {allow_layer2}")
    
    # Create a new Gurobi model for before-layer optimization
    model_before = gp.Model("before_layer3")
    model_before.setParam("OutputFlag", 0)
    
    # Weight change variables
    dw = {}
    # Layer 1→2 (w1, w2, w3, w4)
    for i in range(1, 5):
        dw[i] = model_before.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"dw{i}"
        )
    # Layer 2→3 (w5, w6, w7, w8)
    for i in range(5, 9):
        dw[i] = model_before.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"dw{i}"
        )
    
    # Variables for layer 2 pre-activations
    new_z2_1 = model_before.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="new_z2_1"
    )
    new_z2_2 = model_before.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="new_z2_2"
    )
    
    # Variables for layer 2 post-activations
    new_v2_1 = model_before.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="new_v2_1")
    new_v2_2 = model_before.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="new_v2_2")
    
    # Variables for layer 3 pre-activations
    new_z3_1 = model_before.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="new_z3_1"
    )
    new_z3_2 = model_before.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="new_z3_2"
    )
    
    model_before.update()
    
    # Constraints for layer 1→2 with weight changes
    # Base weights: [[0.5, -0.3], [0.8, 0.2]]
    model_before.addConstr(
        new_z2_1 == (0.5 + dw[1]) * input_val_1 + (-0.3 + dw[2]) * input_val_2,
        name="new_z2_1_calc"
    )
    model_before.addConstr(
        new_z2_2 == (0.8 + dw[3]) * input_val_1 + (0.2 + dw[4]) * input_val_2,
        name="new_z2_2_calc"
    )
    
    # ReLU activations
    model_before.addGenConstrMax(new_v2_1, [new_z2_1, 0.0], name="relu_new_2_1")
    model_before.addGenConstrMax(new_v2_2, [new_z2_2, 0.0], name="relu_new_2_2")
    
    # Constraints for layer 2→3 with weight changes
    # Base weights: [[1.2, -0.7], [0.4, 1.5]]
    model_before.addConstr(
        new_z3_1 == (1.2 + dw[5]) * new_v2_1 + (-0.7 + dw[6]) * new_v2_2,
        name="new_z3_1_calc"
    )
    model_before.addConstr(
        new_z3_2 == (0.4 + dw[7]) * new_v2_1 + (1.5 + dw[8]) * new_v2_2,
        name="new_z3_2_calc"
    )
    
    # Fix weights based on boolean flags
    if not allow_layer1:
        for i in range(1, 5):
            model_before.addConstr(dw[i] == 0, name=f"fix_dw{i}")
    if not allow_layer2:
        for i in range(5, 9):
            model_before.addConstr(dw[i] == 0, name=f"fix_dw{i}")
    
    # Target constraints: new_z3 should match perturbed_z3
    if perturbed_z3_1 <= 0:
        model_before.addConstr(new_z3_1 <= 0, name="target_z3_1")
    else:
        model_before.addConstr(new_z3_1 == perturbed_z3_1, name="target_z3_1")
    
    if perturbed_z3_2 <= 0:
        model_before.addConstr(new_z3_2 <= 0, name="target_z3_2")
    else:
        model_before.addConstr(new_z3_2 == perturbed_z3_2, name="target_z3_2")
    
    # Objective: minimize maximum absolute weight change
    max_abs_before = model_before.addVar(
        lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="max_abs_before"
    )
    
    for i in range(1, 9):
        model_before.addConstr(max_abs_before >= dw[i], name=f"max_abs_before_pos_{i}")
        model_before.addConstr(max_abs_before >= -dw[i], name=f"max_abs_before_neg_{i}")

    model_before.setObjective(max_abs_before, GRB.MINIMIZE)
    model_before.optimize()
    
    if model_before.status == GRB.OPTIMAL:
        print(f"\n  Optimization successful!")
        print(f"\n  Weight modifications (Before Layer 3):")
        delta_weights_before = {}
        print(f"    Layer 1→2:")
        for i in range(1, 5):
            delta_weights_before[i] = dw[i].X
            print(f"      Δw{i} = {dw[i].X:.6f}")
        print(f"    Layer 2→3:")
        for i in range(5, 9):
            delta_weights_before[i] = dw[i].X
            print(f"      Δw{i} = {dw[i].X:.6f}")
        
        max_weight_change_before = max_abs_before.X
        print(f"\n  Maximum absolute weight change (Before Layer 3): {max_weight_change_before:.6f}")
        
        print(f"\n  Verification:")
        print(f"    Achieved z3_1 = {new_z3_1.X:.6f} (target: {perturbed_z3_1:.6f})")
        print(f"    Achieved z3_2 = {new_z3_2.X:.6f} (target: {perturbed_z3_2:.6f})")
    else:
        print(f"\n  Optimization failed with status: {model_before.status}")
        delta_weights_before = None
        max_weight_change_before = None

# Step 5b: Single Layer Modification - After Layer 3 (Layers 3→4 and 4→5) using Gurobi
    print(f"\nStep 5b: Single Layer Modification - After Layer 3 (Using Gurobi)")
    print(f"  Goal: Modify weights in layers after layer 3 to achieve target output with perturbed z3")
    print(f"  Allowed modifications: Layer3→4: {allow_layer3}, Layer4→5: {allow_layer4}")
    
    # Create a new Gurobi model for after-layer optimization
    model_after = gp.Model("after_layer3")
    model_after.setParam("OutputFlag", 0)
    
    # Weight change variables
    dw_after = {}
    # Layer 3→4 (w9, w10, w11, w12)
    for i in range(9, 13):
        dw_after[i] = model_after.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"dw{i}"
        )
    # Layer 4→5 (w13, w14, w15, w16)
    for i in range(13, 17):
        dw_after[i] = model_after.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"dw{i}"
        )
    
    # Fixed input to this subnetwork (perturbed v3 values)
    perturbed_v3_1 = max(0, perturbed_z3_1)
    perturbed_v3_2 = max(0, perturbed_z3_2)
    
    # Variables for layer 4 pre-activations
    new_z4_1 = model_after.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="new_z4_1"
    )
    new_z4_2 = model_after.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="new_z4_2"
    )
    
    # Variables for layer 4 post-activations
    new_v4_1 = model_after.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="new_v4_1")
    new_v4_2 = model_after.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="new_v4_2")
    
    # Variables for layer 5 (output)
    new_v5_1 = model_after.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="new_v5_1"
    )
    new_v5_2 = model_after.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="new_v5_2"
    )
    
    model_after.update()
    
    # Constraints for layer 3→4 with weight changes
    # Base weights: [[-0.5, 0.9], [1.1, -0.4]]
    model_after.addConstr(
        new_z4_1 == (-0.5 + dw_after[9]) * perturbed_v3_1 + (0.9 + dw_after[10]) * perturbed_v3_2,
        name="new_z4_1_calc"
    )
    model_after.addConstr(
        new_z4_2 == (1.1 + dw_after[11]) * perturbed_v3_1 + (-0.4 + dw_after[12]) * perturbed_v3_2,
        name="new_z4_2_calc"
    )
    
    # ReLU activations
    model_after.addGenConstrMax(new_v4_1, [new_z4_1, 0.0], name="relu_new_4_1")
    model_after.addGenConstrMax(new_v4_2, [new_z4_2, 0.0], name="relu_new_4_2")
    
    # Constraints for layer 4→5 with weight changes
    # Base weights: [[0.6, -1.0], [-0.8, 0.7]]
    model_after.addConstr(
        new_v5_1 == (0.6 + dw_after[13]) * new_v4_1 + (-1.0 + dw_after[14]) * new_v4_2,
        name="new_v5_1_calc"
    )
    model_after.addConstr(
        new_v5_2 == (-0.8 + dw_after[15]) * new_v4_1 + (0.7 + dw_after[16]) * new_v4_2,
        name="new_v5_2_calc"
    )
    
    # Fix weights based on boolean flags
    if not allow_layer3:
        for i in range(9, 13):
            model_after.addConstr(dw_after[i] == 0, name=f"fix_dw{i}")
    if not allow_layer4:
        for i in range(13, 17):
            model_after.addConstr(dw_after[i] == 0, name=f"fix_dw{i}")
    
    # Target constraints: new output should match target output
    model_after.addConstr(new_v5_1 == target_output_1, name="target_v5_1")
    model_after.addConstr(new_v5_2 == target_output_2, name="target_v5_2")
    
    # Objective: minimize maximum absolute weight change
    max_abs_after = model_after.addVar(
        lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="max_abs_after"
    )
    
    for i in range(9, 17):
        model_after.addConstr(max_abs_after >= dw_after[i], name=f"max_abs_after_pos_{i}")
        model_after.addConstr(max_abs_after >= -dw_after[i], name=f"max_abs_after_neg_{i}")
    
    model_after.setObjective(max_abs_after, GRB.MINIMIZE)
    model_after.optimize()
    
    if model_after.status == GRB.OPTIMAL:
        print(f"\n  Optimization successful!")
        print(f"\n  Input to modified network (perturbed v3):")
        print(f"    v3_1 = {perturbed_v3_1:.6f}")
        print(f"    v3_2 = {perturbed_v3_2:.6f}")
        
        print(f"\n  Weight modifications (After Layer 3):")
        delta_weights_after = {}
        print(f"    Layer 3→4:")
        for i in range(9, 13):
            delta_weights_after[i] = dw_after[i].X
            print(f"      Δw{i} = {dw_after[i].X:.6f}")
        print(f"    Layer 4→5:")
        for i in range(13, 17):
            delta_weights_after[i] = dw_after[i].X
            print(f"      Δw{i} = {dw_after[i].X:.6f}")
        
        max_weight_change_after = max_abs_after.X
        print(f"\n  Maximum absolute weight change (After Layer 3): {max_weight_change_after:.6f}")
        
        print(f"\n  Verification:")
        print(f"    Achieved v5_1 = {new_v5_1.X:.6f} (target: {target_output_1:.6f})")
        print(f"    Achieved v5_2 = {new_v5_2.X:.6f} (target: {target_output_2:.6f})")
    else:
        print(f"\n  Optimization failed with status: {model_after.status}")
        delta_weights_after = None
        max_weight_change_after = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if max_weight_change_before is not None:
        print(f"Maximum weight change (Before Layer 3): {max_weight_change_before:.6f}")
    if max_weight_change_after is not None:
        print(f"Maximum weight change (After Layer 3): {max_weight_change_after:.6f}")
    print("=" * 60)

    return {
        "current_output": (current_output_1, current_output_2),
        "target_output": (target_output_1, target_output_2),
        "current_z3": (current_z3_1, current_z3_2),
        "perturbed_z3": (perturbed_z3_1, perturbed_z3_2),
        "gradients_z3": (grad_z3_1, grad_z3_2),
        "alpha": alpha,
        "delta_weights_before": delta_weights_before,
        "delta_weights_after": delta_weights_after,
        "max_weight_change_before": max_weight_change_before,
        "max_weight_change_after": max_weight_change_after,
    }


# Example usage
if __name__ == "__main__":
    # Create the neural network
    model, variables = create_neural_network()

    find_optimal_weights(model, variables)

    # gradient_based_update(model, variables)
    # gradient_based_update_multi(model, variables)

    sys.exit()

    # Example: Forward pass with specific inputs and weights
    input_values = [1.0, 2.0]  # v1,1 = 1.0, v1,2 = 2.0
    weight_values = {
        1: 0.5,  # w1
        2: -0.3,  # w2
        3: 0.2,  # w3
        4: 0.8,  # w4
        5: 0.1,  # w5
        6: -0.2,  # w6
        7: 0.4,  # w7
        8: 0.6,  # w8
    }

    print("=" * 60)
    print("Neural Network Forward Pass")
    print("=" * 60)
    print(f"\nInput values: v1,1 = {input_values[0]}, v1,2 = {input_values[1]}")
    print(f"\nWeights:")
    for i, val in weight_values.items():
        print(f"  w{i} = {val}")

    results = forward_pass(model, variables, input_values, weight_values)

    if results:
        print(f"\n{'=' * 60}")
        print("Results:")
        print("=" * 60)
        print(f"\nHidden layer (after ReLU):")
        print(f"  v2,1 = {results['hidden'][0]:.4f}")
        print(f"  v2,2 = {results['hidden'][1]:.4f}")
        print(f"\nOutput layer (after ReLU):")
        print(f"  v3,1 = {results['output'][0]:.4f}")
        print(f"  v3,2 = {results['output'][1]:.4f}")
        print("=" * 60)
