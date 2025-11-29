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
    w = {}
    for i in range(1, 11):
        w[i] = model.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"w{i}"
        )

    # Layer 1 (input layer)
    v1_1 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v1_1"
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

    # Layer 1 to Layer 2 connections (weights: 1, 1)
    model.addConstr(z2_1 == (1 + w[1]) * v1_1, name="z2_1_calc")
    model.addConstr(z2_2 == (1 + w[2]) * v1_1, name="z2_2_calc")
    model.addGenConstrMax(v2_1, [z2_1, 0.0], name="relu_2_1")
    model.addGenConstrMax(v2_2, [z2_2, 0.0], name="relu_2_2")

    # Layer 2 to Layer 3 connections (weights: 0.01, 100)
    model.addConstr(z3_1 == (0.01 + w[3]) * v2_1, name="z3_1_calc")
    model.addConstr(z3_2 == (100 + w[4]) * v2_2, name="z3_2_calc")
    model.addGenConstrMax(v3_1, [z3_1, 0.0], name="relu_3_1")
    model.addGenConstrMax(v3_2, [z3_2, 0.0], name="relu_3_2")

    # Layer 3 to Layer 4 connections (weights: 1000, 0.01)
    model.addConstr(z4_1 == (1000 + w[5]) * v3_1, name="z4_1_calc")
    model.addConstr(z4_2 == (0.01 + w[6]) * v3_2, name="z4_2_calc")
    model.addGenConstrMax(v4_1, [z4_1, 0.0], name="relu_4_1")
    model.addGenConstrMax(v4_2, [z4_2, 0.0], name="relu_4_2")

    # Layer 4 to Layer 5 connections (weights: 1, -1 for v4_1; 1, -1 for v4_2)
    model.addConstr(v5_1 == (1 + w[7]) * v4_1 + (1 + w[8]) * v4_2, name="v5_1_calc")
    model.addConstr(v5_2 == (-1 + w[9]) * v4_1 + (-1 + w[10]) * v4_2, name="v5_2_calc")

    return model, {
        "weights": w,
        "layer1": (v1_1,),
        "layer2": (v2_1, v2_2),
        "layer3": (v3_1, v3_2),
        "layer4": (v4_1, v4_2),
        "layer5": (v5_1, v5_2),
        "layer2_pre_activation": (z2_1, z2_2),
        "layer3_pre_activation": (z3_1, z3_2),
        "layer4_pre_activation": (z4_1, z4_2),
    }


def forward_pass(model, variables, input_values, weight_values):
    """
    Perform a forward pass through the network with given inputs and weights.

    Args:
        model: Gurobi model
        variables: Dictionary of model variables
        input_values: List/tuple of input values [v1_1, v1_2]
        weight_values: Dictionary of weight values {1: w1, 2: w2, ..., 8: w8}

    Returns:
        Dictionary with values of all nodes
    """
    # Fix input values
    v1_1, v1_2 = variables["layer1"]
    model.addConstr(v1_1 == input_values[0], name="input_1_fixed")
    model.addConstr(v1_2 == input_values[1], name="input_2_fixed")

    # Fix weight values
    for i, val in weight_values.items():
        model.addConstr(variables["weights"][i] == val, name=f"weight_{i}_fixed")

    # Optimize (no objective needed for forward pass, just feasibility)
    model.setObjective(0, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        results = {
            "input": (v1_1.X, v1_2.X),
            "hidden": (variables["layer2"][0].X, variables["layer2"][1].X),
            "output": (variables["layer3"][0].X, variables["layer3"][1].X),
            "weights": {i: variables["weights"][i].X for i in range(1, 9)},
        }
        return results
    else:
        print(f"Optimization failed with status: {model.status}")
        return None


def find_optimal_weights(model, variables):
    print(variables)

    # Fix input value to 1
    v1_1 = variables["layer1"][0]
    model.addConstr(v1_1 == 1.0, name="input_1_fixed_opt")

    # Add constraint that output_2 > output_1
    v5_1, v5_2 = variables["layer5"]
    model.addConstr(v5_2 >= v5_1 + 0.0001, name="output_2_greater_than_output_1")

    # Create auxiliary variable for the maximum absolute value of all weights
    # max_abs = model.addVar(
    #     lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="max_abs_weight"
    # )
    # # Add constraints: max_abs >= |wi| for all weights (1 to 10)
    # for i in range(1, 11):
    #     model.addConstr(max_abs >= variables["weights"][i], name=f"max_abs_pos_{i}")
    #     model.addConstr(max_abs >= -variables["weights"][i], name=f"max_abs_neg_{i}")

    abs_weights = {}
    for i in range(1, 11):
        abs_weights[i] = model.addVar(
            lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"abs_w{i}"
        )
        # Add constraints: abs_weights[i] >= |w[i]|
        model.addConstr(abs_weights[i] >= variables["weights"][i], name=f"abs_pos_{i}")
        model.addConstr(abs_weights[i] >= -variables["weights"][i], name=f"abs_neg_{i}")

    for i in range(1, 1):
        model.addConstr(variables["weights"][i] == 0, name=f"fix_weights_{i}")

    # Set objective: minimize the maximum absolute value
    # model.setObjective(max_abs, GRB.MINIMIZE)
    total_change = sum(abs_weights[i] for i in range(1, 11))
    model.setObjective(total_change, GRB.MINIMIZE)

    # Optimize
    model.setParam("OutputFlag", 1)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("\n" + "=" * 60)
        print("OPTIMAL SOLUTION FOUND")
        print("=" * 60)
        print(f"\nInput value: v1_1 = {v1_1.X:.4f}")

        print(f"\nAll weights:")
        total_sum = 0
        for i in range(1, 11):
            print(f"  w{i} = {variables['weights'][i].X:.6f}")
            total_sum += abs(variables["weights"][i].X)

        print(f"\nTotal sum of absolute weight changes: {total_sum:.6f}")

        # Get pre-activation variables
        z2_1, z2_2 = variables["layer2_pre_activation"]
        z3_1, z3_2 = variables["layer3_pre_activation"]
        z4_1, z4_2 = variables["layer4_pre_activation"]

        print(f"\nLayer 2 pre-activations:")
        print(f"  z2_1 = {z2_1.X:.6f}")
        print(f"  z2_2 = {z2_2.X:.6f}")

        print(f"\nLayer 2 post-activations:")
        print(f"  v2_1 = {variables['layer2'][0].X:.6f}")
        print(f"  v2_2 = {variables['layer2'][1].X:.6f}")

        print(f"\nLayer 3 pre-activations:")
        print(f"  z3_1 = {z3_1.X:.6f}")
        print(f"  z3_2 = {z3_2.X:.6f}")

        print(f"\nLayer 3 post-activations:")
        print(f"  v3_1 = {variables['layer3'][0].X:.6f}")
        print(f"  v3_2 = {variables['layer3'][1].X:.6f}")

        print(f"\nLayer 4 pre-activations:")
        print(f"  z4_1 = {z4_1.X:.6f}")
        print(f"  z4_2 = {z4_2.X:.6f}")

        print(f"\nLayer 4 post-activations:")
        print(f"  v4_1 = {variables['layer4'][0].X:.6f}")
        print(f"  v4_2 = {variables['layer4'][1].X:.6f}")

        print(f"\nOutput layer (Layer 5):")
        print(f"  v5_1 (output_1) = {v5_1.X:.6f}")
        print(f"  v5_2 (output_2) = {v5_2.X:.6f}")
        print(f"  Difference (v5_2 - v5_1) = {v5_2.X - v5_1.X:.6f}")
        print("=" * 60)
    else:
        print(f"Optimization failed with status: {model.status}")


def gradient_based_update(model, variables):
    # Fix input value
    v1_1 = variables["layer1"][0]
    input_val_1 = 1.0

    model.addConstr(v1_1 == input_val_1, name="input_1_grad_fixed")

    # Fix all weights to 0 (use base network)
    for i in range(1, 11):
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

    # Get current weights
    current_weights = {i: variables["weights"][i].X for i in range(1, 11)}

    print("\n" + "=" * 60)
    print("GRADIENT-BASED UPDATE")
    print("=" * 60)
    print(f"\nStep 1: Current Forward Pass")
    print(f"  Input: v1_1 = {input_val_1}")

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
        f"  Constraint satisfied? v5_2 >= v5_1: {current_output_2 >= current_output_1}"
    )

    # Step 2: Find nearest output satisfying constraint (minimum MSE)
    # Create new variables for target output
    target_v5_1 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="target_v5_1"
    )
    target_v5_2 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="target_v5_2"
    )

    # Add constraint: target_v5_2 >= target_v5_1
    model.addConstr(target_v5_2 >= target_v5_1, name="target_constraint")

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
    print(f"  Constraint satisfied? v5_2 >= v5_1: {target_output_2 >= target_output_1}")

    # Step 3: Calculate gradients using PyTorch
    print(f"\nStep 3: Gradient Calculation (PyTorch)")

    # Layer 2 gradients
    z2_1_tensor = torch.tensor(current_z2_1, requires_grad=True)
    z2_2_tensor = torch.tensor(current_z2_2, requires_grad=True)
    v2_1_tensor = torch.relu(z2_1_tensor)
    v2_2_tensor = torch.relu(z2_2_tensor)

    # Forward pass through remaining layers
    # Layer 3
    w3, w4 = current_weights[3], current_weights[4]
    z3_1_tensor = (0.01 + w3) * v2_1_tensor
    z3_2_tensor = (100 + w4) * v2_2_tensor
    v3_1_tensor = torch.relu(z3_1_tensor)
    v3_2_tensor = torch.relu(z3_2_tensor)

    # Layer 4
    w5, w6 = current_weights[5], current_weights[6]
    z4_1_tensor = (1000 + w5) * v3_1_tensor
    z4_2_tensor = (0.01 + w6) * v3_2_tensor
    v4_1_tensor = torch.relu(z4_1_tensor)
    v4_2_tensor = torch.relu(z4_2_tensor)

    # Layer 5 (output)
    w7, w8, w9, w10 = (
        current_weights[7],
        current_weights[8],
        current_weights[9],
        current_weights[10],
    )
    v5_1_tensor = (1 + w7) * v4_1_tensor + (1 + w8) * v4_2_tensor
    v5_2_tensor = (-1 + w9) * v4_1_tensor + (-1 + w10) * v4_2_tensor

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
    z4_1_tensor = (1000 + w5) * v3_1_tensor
    z4_2_tensor = (0.01 + w6) * v3_2_tensor
    v4_1_tensor = torch.relu(z4_1_tensor)
    v4_2_tensor = torch.relu(z4_2_tensor)

    v5_1_tensor = (1 + w7) * v4_1_tensor + (1 + w8) * v4_2_tensor
    v5_2_tensor = (-1 + w9) * v4_1_tensor + (-1 + w10) * v4_2_tensor

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

    v5_1_tensor = (1 + w7) * v4_1_tensor + (1 + w8) * v4_2_tensor
    v5_2_tensor = (-1 + w9) * v4_1_tensor + (-1 + w10) * v4_2_tensor

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


# Example usage
if __name__ == "__main__":
    # Create the neural network
    model, variables = create_neural_network()

    # find_optimal_weights(model, variables)

    gradient_based_update(model, variables)
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
