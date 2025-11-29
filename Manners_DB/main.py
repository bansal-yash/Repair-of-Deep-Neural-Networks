import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from gurobipy import Model, GRB
import gurobipy as gb
import Neural_Net
import encode_nn

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def read_data(filename):
    data = pd.read_csv(filename)
    # data = data.groupby("Stamp").mean()
    data = data.drop(columns=["Stamp", "Using arrow"])
    data = data.rename(columns={"Using circle": "operating mode"})
    data.iloc[:, -8:] = data.iloc[:, -8:] / 5

    # print(data)
    return data


if __name__ == "__main__":
    train_data = read_data("all_data.csv")
    test_data = read_data("data_test.csv")

    x_train = train_data.iloc[:, :28].values
    y_train = train_data.iloc[:, 28:].values
    x_test = test_data.iloc[:, :28].values
    y_test = test_data.iloc[:, 28:].values

    print(x_train.shape)
    print(y_train.shape)

    x_train_new = x_train[:8840]
    y_train_new = y_train[:8840]
    x_test_new = x_train[8840:9945]
    y_test_new = y_train[8840:9945]

    x_train = x_train_new
    y_train = y_train_new
    x_test = x_test_new
    y_test = y_test_new

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # print(x_train_tensor)
    # print(x_test_tensor)
    # print(y_train_tensor)
    # print(y_test_tensor)

    total_samples_added = 0

    while True:
        input_size = x_train_tensor.shape[1]
        output_size = y_train_tensor.shape[1]

        batch_size = 100
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        hidden_size = 6
        num_hidden_layers = 6
        lr = 0.001

        model = Neural_Net.NeuralNet(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        Neural_Net.train_model(
            model, train_loader, test_loader, criterion, optimizer, num_epochs=100
        )

        # break

        m, variables = encode_nn.encode_nn(
            num_inputs=input_size,
            num_outputs=output_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            model=model,
        )

        for j in range(input_size):
            m.addConstr(variables["x"][j] >= 0)
            m.addConstr(variables["x"][j] <= 1)

        m.addConstr(variables["x"][22] <= 0.1)

        epsilon = 1e-4

        y_min = m.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y_min"
        )
        m.addConstr(
            y_min
            == gb.min_(
                variables["y"][0],
                variables["y"][2],
                variables["y"][3],
                variables["y"][4],
                variables["y"][5],
                variables["y"][6],
                variables["y"][7],
            ),
            name="min",
        )
        m.addConstr(variables["y"][1] >= y_min + epsilon)  # y > some other actions

        m.update()

        m.setParam(
            GRB.Param.PoolSearchMode, 1
        )  # Active search for multiple feasible solutions
        m.setParam(GRB.Param.PoolSolutions, 50)  # Limit to 50 diverse solutions
        # m.setParam(GRB.Param.PoolGap, 1)

        m.optimize()

        binary_inputs = [1, 6, 7, 16, 17, 18, 19, 20, 21, 24, 27]

        if m.SolCount > 0:
            print(f"SATISFIABLE: {m.SolCount} solutions found\n")
            for i in range(m.SolCount):
                # print(f"\n--- Solution {i + 1} ---")
                m.setParam(GRB.Param.SolutionNumber, i)

                x_sat = []
                y_sat = []

                for var in variables["x"]:
                    # print(f"{var.VarName} = {var.Xn}")
                    x_sat.append(var.Xn)
                for var in variables["y"]:
                    # print(f"{var.VarName} = {var.Xn}")
                    y_sat.append(var.Xn)

                m2 = Model("second_optimization")
                m2.setParam("OutputFlag", 0)

                s = [
                    m2.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"s_{i+1}")
                    for i in range(8)
                ]

                m2.update()

                for i in range(8):
                    m2.addConstr(
                        (y_sat[1]) + s[1] <= (y_sat[i]) + s[i], name=f"c_{i+1}"
                    )

                obj = gb.quicksum(s[i] * s[i] for i in range(8))
                m2.setObjective(obj, GRB.MINIMIZE)

                m2.optimize()

                if m2.status == GRB.OPTIMAL:
                    # print("Optimal s values:")
                    s_opt = [var.X for var in s]
                    # print(s_opt)
                else:
                    print("No optimal solution found.")

                y_new = [y_sat[i] + s_opt[i] for i in range(8)]

                x_sam = torch.tensor(x_sat, dtype=torch.float32)
                x_sam = x_sam.unsqueeze(0)

                y_new = torch.tensor(y_new, dtype=torch.float32)

                y_sam = y_new.unsqueeze(0)
                x_train_tensor = torch.cat([x_train_tensor, x_sam], dim=0)
                y_train_tensor = torch.cat([y_train_tensor, y_sam], dim=0)

                num_samples_added = 1
                while num_samples_added < 20:
                    sample = 0.1 * torch.randn(8)
                    y_sam = y_new + sample

                    sample_x_noise = 0.1 * torch.randn(28)
                    x_sam = []

                    for temp1 in range(1, 29):
                        if temp1 in binary_inputs:
                            x_sam.append(x_sat[temp1 - 1])
                        else:
                            x_sam.append(x_sat[temp1 - 1] + sample_x_noise[temp1 - 1])

                    x_sam = torch.tensor(x_sam, dtype=torch.float32)

                    if (
                        y_sam[1]
                        <= min(y_sam)
                        # and (min(x_sam) >= 0)
                        # and (max(x_sam) <= 1)
                        # and (x_sam[22] <= 0.1)
                    ):
                        y_sam = y_sam.unsqueeze(0)
                        x_sam = x_sam.unsqueeze(0)
                        x_train_tensor = torch.cat([x_train_tensor, x_sam], dim=0)
                        y_train_tensor = torch.cat([y_train_tensor, y_sam], dim=0)

                        num_samples_added += 1

                total_samples_added += num_samples_added
        else:
            print("UNSATISFIABLE")
            break

        print(total_samples_added)

        example_input = torch.randn(1, input_size)

        # Export the model to ONNX
        torch.onnx.export(
            model,                      # model being run
            example_input,              # example input tensor
            "trained_model.onnx",       # output file name
            export_params=True,         # store trained weights inside the model file
            opset_version=12,           # ONNX opset version
            do_constant_folding=True,   # optimize constant expressions
            input_names=['input'],      # name for the model's input
            output_names=['output'],    # name for the model's output
            dynamic_axes={
                'input': {0: 'batch_size'},   # allow variable batch size
                'output': {0: 'batch_size'}
            }
        )

        print("✅ Model successfully saved as trained_model.onnx")

        if (m.SolCount == 1):
            break
    
        break

    print(total_samples_added)
