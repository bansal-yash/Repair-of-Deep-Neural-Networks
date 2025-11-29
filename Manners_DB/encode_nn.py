import Neural_Net
from gurobipy import Model, GRB
import gurobipy as gb


def encode_nn(
    num_inputs, num_outputs, hidden_size, num_hidden_layers, model: Neural_Net.NeuralNet
):
    m = Model("nn_encoding")
    m.setParam("OutputFlag", 0)  # Silence solver output

    num_total_layers = num_hidden_layers + 2
    v = {}

    binary_inputs = [1, 6, 7, 16, 17, 18, 19, 20, 21, 24, 27]

    # Input layer
    v["x"] = [
        m.addVar(
            lb=0.0,
            ub=1.0,
            vtype=GRB.BINARY if j in binary_inputs else GRB.CONTINUOUS,
            name=f"x_{j}",
        )
        for j in range(1, num_inputs + 1)
    ]

    v["v_1_out"] = v["x"]  # Direct connection

    # Hidden layers
    for i in range(2, num_total_layers):
        v[f"v_{i}_in"] = [
            m.addVar(
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name=f"v_{i}_{j}_in",
            )
            for j in range(1, hidden_size + 1)
        ]
        v[f"v_{i}_out"] = [
            m.addVar(
                lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"v_{i}_{j}_out"
            )
            for j in range(1, hidden_size + 1)
        ]

    # Output layer
    v[f"v_{num_total_layers}_in"] = [
        m.addVar(
            lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"v_{num_total_layers}_{j}_in"
        )
        for j in range(1, num_outputs + 1)
    ]
    v["y"] = [
        m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_{j}")
        for j in range(1, num_outputs + 1)
    ]

    m.update()

    # Load parameters
    params_dict = {
        name: param.detach().numpy() for name, param in model.state_dict().items()
    }

    # Hidden layer constraints
    for i in range(2, num_total_layers):
        for j in range(hidden_size):
            if i == 2:
                bias = float(params_dict["input_layer.bias"][j])
                weights = params_dict["input_layer.weight"][j]
                prev_layer = v["v_1_out"]
            else:
                bias = float(params_dict[f"hidden_layers.{2*i-5}.bias"][j])
                weights = params_dict[f"hidden_layers.{2*i-5}.weight"][j]
                prev_layer = v[f"v_{i - 1}_out"]

            v_in = v[f"v_{i}_in"][j]
            v_out = v[f"v_{i}_out"][j]

            m.addConstr(
                v_in
                == sum(weights[k] * prev_layer[k] for k in range(len(weights))) + bias
            )

            m.addConstr(v_out == gb.max_(v_in, 0.0))

    # Output layer constraints
    for j in range(num_outputs):
        bias = float(params_dict["output_layer.bias"][j])
        weights = params_dict["output_layer.weight"][j]
        prev_layer = v[f"v_{num_total_layers - 1}_out"]
        v_in = v[f"v_{num_total_layers}_in"][j]

        m.addConstr(
            v_in == sum(weights[k] * prev_layer[k] for k in range(len(weights))) + bias
        )
        m.addConstr(v["y"][j] == v_in)

    return m, v
