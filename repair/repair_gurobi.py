import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sys
import torch
import torch.nn as nn

def create_neural_network():
    """
    Implements the neural network from the diagram:
    - Layer 1 (V1): 2 input nodes (v1,1 and v1,2)
    - Layer 2 (V2): 2 hidden nodes (v2,1 and v2,2)
    - Layer 3 (V3): 2 output nodes (v3,1 and v3,2)

    Connections with weights:
    V1 to V2:
    - v1,1 to v2,1: 1 + w1
    - v1,1 to v2,2: 2 + w3
    - v1,2 to v2,2: 2 + w2
    - v1,2 to v2,2: -1 + w4

    V2 to V3:
    - v2,1 to v3,1: 1 + w5
    - v2,1 to v3,2: -1 + w6
    - v2,2 to v3,1: 1 + w7
    - v2,2 to v3,2: 1 + w8
    """

    # Create a new model
    model = gp.Model("neural_network")

    # Suppress output (set to 1 to see solver output)
    model.setParam("OutputFlag", 0)

    # Define weight variables (can be positive or negative)
    w = {}
    for i in range(1, 9):
        w[i] = model.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"w{i}"
        )

    # Define node variables for each layer
    # Layer 1 (input layer)
    v1_1 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v1_1"
    )
    v1_2 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v1_2"
    )

    # Layer 2 (hidden layer) - before activation
    z2_1 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z2_1"
    )
    z2_2 = model.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z2_2"
    )

    # Layer 2 (hidden layer) - after ReLU activation
    v2_1 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v2_1")
    v2_2 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v2_2")

    # Layer 3 (output layer) - before activation
    # z3_1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
    #                     vtype=GRB.CONTINUOUS, name="z3_1")
    # z3_2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
    #                     vtype=GRB.CONTINUOUS, name="z3_2")

    # Layer 3 (output layer) - after ReLU activation
    v3_1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v3_1")
    v3_2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v3_2")

    # Update model to integrate new variables
    model.update()

    # Layer 1 to Layer 2 connections
    # z2_1 = (1 + w1) * v1_1 + (2 + w3) * v1_2
    model.addConstr(z2_1 == (1 + w[1]) * v1_1 + (-2 + w[3]) * v1_2, name="z2_1_calc")

    # z2_2 = (2 + w2) * v1_1 + (-1 + w4) * v1_2
    model.addConstr(z2_2 == (2 + w[2]) * v1_1 + (-1 + w[4]) * v1_2, name="z2_2_calc")

    # ReLU activations for layer 2 (using max formulation)
    model.addGenConstrMax(v2_1, [z2_1, 0.0], name="relu_2_1")
    model.addGenConstrMax(v2_2, [z2_2, 0.0], name="relu_2_2")

    # Layer 2 to Layer 3 connections
    # z3_1 = (1 + w5) * v2_1 + (1 + w7) * v2_2
    model.addConstr(v3_1 == (1 + w[5]) * v2_1 + (-1 + w[7]) * v2_2, name="z3_1_calc")

    # z3_2 = (-1 + w6) * v2_1 + (1 + w8) * v2_2
    model.addConstr(v3_2 == (-1 + w[6]) * v2_1 + (1 + w[8]) * v2_2, name="z3_2_calc")

    # # ReLU activations for layer 3
    # model.addGenConstrMax(v3_1, [z3_1, 0.0], name="relu_3_1")
    # model.addGenConstrMax(v3_2, [z3_2, 0.0], name="relu_3_2")

    return model, {
        "weights": w,
        "layer1": (v1_1, v1_2),
        "layer2": (v2_1, v2_2),
        "layer3": (v3_1, v3_2),
        "layer2_pre_activation": (z2_1, z2_2),
        # 'layer3_pre_activation': (z3_1, z3_2)
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
    # sys.exit()

    # Fix input values to 3 and 4
    v1_1, v1_2 = variables["layer1"]
    model.addConstr(v1_1 == 10.0, name="input_1_fixed_opt")
    model.addConstr(v1_2 == 4.0, name="input_2_fixed_opt")

    # Add constraint that output_1 > output_2
    v3_1, v3_2 = variables["layer3"]
    model.addConstr(v3_1 >= v3_2, name="output_1_greater_than_output_2")

    # Create auxiliary variable for the maximum absolute value of all weights
    max_abs = model.addVar(
        lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="max_abs_weight"
    )

    # Add constraints: max_abs >= |wi| for all weights
    for i in range(1, 9):
        model.addConstr(max_abs >= variables["weights"][i], name=f"max_abs_pos_{i}")
        model.addConstr(max_abs >= -variables["weights"][i], name=f"max_abs_neg_{i}")

    # for i in range(1, 9):
    #     model.addConstr(variables["weights"][i] == 0, name=f"fix_weights_{i}")

    # Set objective: minimize the maximum absolute value
    model.setObjective(max_abs, GRB.MINIMIZE)

    # Optimize
    model.setParam("OutputFlag", 1)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("\n" + "=" * 60)
        print("OPTIMAL SOLUTION FOUND")
        print("=" * 60)
        print(f"\nInput values: v1,1 = {v1_1.X:.4f}, v1,2 = {v1_2.X:.4f}")
        print(f"\nAll weights:")
        for i in range(1, 9):
            print(f"  w{i} = {variables['weights'][i].X:.6f}")
        print(f"\nWeights of interest (w6, w7, w8):")
        print(f"  w6 = {variables['weights'][6].X:.6f}")
        print(f"  w7 = {variables['weights'][7].X:.6f}")
        print(f"  w8 = {variables['weights'][8].X:.6f}")
        print(f"\nMaximum absolute weight value: {max_abs.X:.6f}")
        print(f"\nHidden layer outputs:")
        print(f"  v2,1 = {variables['layer2'][0].X:.6f}")
        print(f"  v2,2 = {variables['layer2'][1].X:.6f}")
        print(f"\nOutput layer:")
        print(f"  v3,1 (output_1) = {variables['layer3'][0].X:.6f}")
        print(f"  v3,2 (output_2) = {variables['layer3'][1].X:.6f}")
        print(
            f"  Difference (v3,1 - v3,2) = {variables['layer3'][0].X - variables['layer3'][1].X:.6f}"
        )
        print("=" * 60)
    else:
        print(f"Optimization failed with status: {model.status}")


def gradient_based_update(model, variables):
    # Fix input values
    v1_1, v1_2 = variables["layer1"]
    input_val_1 = 10.0
    input_val_2 = 4.0
    
    model.addConstr(v1_1 == input_val_1, name="input_1_grad_fixed")
    model.addConstr(v1_2 == input_val_2, name="input_2_grad_fixed")
    
    for i in range(1, 9):
        model.addConstr(variables["weights"][i] == 0, name=f"fix_weights_{i}")

    # Step 1: Calculate current output (forward pass with current weights)
    model.setObjective(0, GRB.MINIMIZE)
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        print(f"Forward pass failed with status: {model.status}")
        return None
    
    # Get current output values
    current_output_1 = variables["layer3"][0].X
    current_output_2 = variables["layer3"][1].X
    current_z2_1 = variables["layer2_pre_activation"][0].X
    current_z2_2 = variables["layer2_pre_activation"][1].X
    current_v2_1 = variables["layer2"][0].X
    current_v2_2 = variables["layer2"][1].X
    
    # Get current weights
    current_weights = {i: variables["weights"][i].X for i in range(1, 9)}
    
    print("\n" + "=" * 60)
    print("GRADIENT-BASED UPDATE")
    print("=" * 60)
    print(f"\nStep 1: Current Forward Pass")
    print(f"  Input: v1_1 = {input_val_1}, v1_2 = {input_val_2}")
    print(f"  Hidden (pre-activation): z2_1 = {current_z2_1:.6f}, z2_2 = {current_z2_2:.6f}")
    print(f"  Hidden (post-activation): v2_1 = {current_v2_1:.6f}, v2_2 = {current_v2_2:.6f}")
    print(f"  Current Output: v3_1 = {current_output_1:.6f}, v3_2 = {current_output_2:.6f}")
    print(f"  Constraint satisfied? v3_1 >= v3_2: {current_output_1 >= current_output_2}")
    
    # sys.exit()

    # Step 2: Find nearest output satisfying constraint (minimum MSE)
    # Create new variables for target output
    target_v3_1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                               vtype=GRB.CONTINUOUS, name="target_v3_1")
    target_v3_2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                               vtype=GRB.CONTINUOUS, name="target_v3_2")
    
    # Add constraint: target_v3_1 >= target_v3_2
    model.addConstr(target_v3_1 >= target_v3_2, name="target_constraint")
    
    # Minimize MSE: (target_v3_1 - current_v3_1)^2 + (target_v3_2 - current_v3_2)^2
    diff_1 = target_v3_1 - current_output_1
    diff_2 = target_v3_2 - current_output_2
    
    model.setObjective(diff_1 * diff_1 + diff_2 * diff_2, GRB.MINIMIZE)
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        print(f"Target optimization failed with status: {model.status}")
        return None
    
    target_output_1 = target_v3_1.X
    target_output_2 = target_v3_2.X
    mse = (target_output_1 - current_output_1)**2 + (target_output_2 - current_output_2)**2
    

    print(f"\nStep 2: Nearest Output Satisfying Constraint")
    print(f"  Target Output: v3_1 = {target_output_1:.6f}, v3_2 = {target_output_2:.6f}")
    print(f"  MSE Loss: {mse:.6f}")
    print(f"  Constraint satisfied? v3_1 >= v3_2: {target_output_1 >= target_output_2}")
    
    # Step 3: Calculate gradients using PyTorch
    # Convert to PyTorch tensors
    z2_1_tensor = torch.tensor(current_z2_1, requires_grad=True)
    z2_2_tensor = torch.tensor(current_z2_2, requires_grad=True)
    
    # Apply ReLU
    v2_1_tensor = torch.relu(z2_1_tensor)
    v2_2_tensor = torch.relu(z2_2_tensor)
    
    # Calculate output (using current weights w5, w6, w7, w8)
    w5, w6, w7, w8 = current_weights[5], current_weights[6], current_weights[7], current_weights[8]
    v3_1_tensor = (1 + w5) * v2_1_tensor + (-1 + w7) * v2_2_tensor
    v3_2_tensor = (-1 + w6) * v2_1_tensor + (1 + w8) * v2_2_tensor
    
    # Calculate MSE loss
    target_v3_1_tensor = torch.tensor(target_output_1)
    target_v3_2_tensor = torch.tensor(target_output_2)
    
    loss = (v3_1_tensor - target_v3_1_tensor)**2 + (v3_2_tensor - target_v3_2_tensor)**2
    
    # Backpropagate
    loss.backward()
    
    # Get gradients
    grad_z2_1 = z2_1_tensor.grad.item()
    grad_z2_2 = z2_2_tensor.grad.item()
    
    print(f"\nStep 3: Gradient Calculation (PyTorch)")
    print(f"  ∂Loss/∂z2_1 = {grad_z2_1:.6f}")
    print(f"  ∂Loss/∂z2_2 = {grad_z2_2:.6f}")
    print("=" * 60)
    
    sys.exit()

    return {
        "current_output": (current_output_1, current_output_2),
        "target_output": (target_output_1, target_output_2),
        "current_z2": (current_z2_1, current_z2_2),
        "current_v2": (current_v2_1, current_v2_2),
        "mse_loss": mse,
        "gradients": {
            "dLoss_dz2_1": grad_z2_1,
            "dLoss_dz2_2": grad_z2_2
        },
        "current_weights": current_weights
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
