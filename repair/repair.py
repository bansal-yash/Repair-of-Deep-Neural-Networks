# ====== modifies_layer_minimal.py ======
# Run in the same environment where onnx, torch, onnx2pytorch, maraboupy are available.

import copy
import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx2pytorch import ConvertModel
from maraboupy import Marabou, MarabouCore
import sys
import csv
from pprint import pprint

# -----------------------
# 1) Utility: build a wrapper that applies deltas to chosen layer
# -----------------------
class DeltaWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, target_layer_path: str):
        """
        base_model : the original PyTorch model (already matching your ONNX architecture)
        target_layer_path : dot-separated module path to the layer we want to allow changes to,
                            e.g. "fc2" or "classifier.6"
        This wrapper assumes the target layer is an nn.Linear (change as needed).
        """
        super().__init__()
        self.base = copy.deepcopy(base_model)  # keep original
        # locate the module & its parameters
        parts = target_layer_path.split(".")
        mod = self.base
        for p in parts[:-1]:
            mod = getattr(mod, p)
        self.layer_parent = mod
        self.layer_name = parts[-1]
        self.target_layer = getattr(self.layer_parent, self.layer_name)

        print(self.target_layer)
        if not isinstance(self.target_layer, nn.Linear):
            raise NotImplementedError("Wrapper currently supports modifying nn.Linear weights only.")
        # flatten shape
        self.w_shape = self.target_layer.weight.shape  # (out, in)
        self.num_weights = int(np.prod(self.w_shape))
        # biases: you may choose to also allow bias deltas; for now we can treat biases similarly
        self.has_bias = self.target_layer.bias is not None
        self.b_shape = self.target_layer.bias.shape if self.has_bias else None
        self.num_bias = int(np.prod(self.b_shape)) if self.has_bias else 0

    def forward(self, x, deltas):
        """
        deltas: a 1D tensor of length num_weights (+ num_bias if enabled)
        We will reshape into weight_delta and optional bias_delta and add to original.
        """
        # deltas = deltas[0]
        # print(deltas.shape)
        # assert deltas.dim() == 1 and deltas.numel() in (self.num_weights, self.num_weights + self.num_bias)
        # reconstruct layer with modified weights
        # w_delta = deltas[:self.num_weights].view(self.w_shape)

        w_delta = deltas.view(self.w_shape)

        # if self.has_bias:
        #     b_delta = deltas[self.num_weights:].view(self.b_shape)
        # else:
        #     b_delta = None

        # temporarily modify the layer's weight & bias for forward
        orig_w = self.target_layer.weight
        orig_b = self.target_layer.bias if self.has_bias else None

        # compute modified linear manually to avoid permanently overwriting params
        # forward through layers up to the target layer
        # We'll rely on PyTorch module internals: we run forward manually using hooks if necessary.
        # Simpler: replace weight & bias tensors in the target linear temporarily (non persistent)
        with torch.no_grad():
            self.target_layer.weight = nn.Parameter(orig_w + w_delta)
            if self.has_bias:
                # self.target_layer.bias = nn.Parameter(orig_b + b_delta)
                self.target_layer.bias = nn.Parameter(orig_b)


        out = self.base(x)

        # restore original params (important)
        with torch.no_grad():
            self.target_layer.weight = orig_w
            if self.has_bias:
                self.target_layer.bias = orig_b

        return out

# -----------------------
# 2) Export wrapper to ONNX with extra 'deltas' input
# -----------------------
def export_with_deltas(pytorch_model, target_layer_path, onnx_outpath="model_with_deltas.onnx"):
    wrapper = DeltaWrapper(pytorch_model, target_layer_path)
    print(wrapper)
    wrapper.eval()
    # dummy inputs: input x and deltas
    dummy_x = torch.randn(1, 2)  # adjust input size to your model input shape
    # dummy_d = torch.zeros(1, wrapper.num_weights + (wrapper.num_bias if wrapper.has_bias else 0))
    dummy_d = torch.zeros(1, wrapper.num_weights)


    print(dummy_x)
    print(dummy_d.shape)
    
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_d),
        onnx_outpath,
        input_names=["input", "deltas"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch"}, "deltas": {0: "batch"}},
    )
    print(f"Exported ONNX with deltas to {onnx_outpath}")
    # return onnx_outpath, wrapper.num_weights + (wrapper.num_bias if wrapper.has_bias else 0)
    return onnx_outpath, wrapper.num_weights

# -----------------------
# 3) Use Marabou to check property for a given M
# -----------------------
def check_with_M(onnx_path, counter_examples, num_deltas, M, property_add_constraints_fn):
    """
    - onnx_path: ONNX file exported from above
    - num_deltas: how many delta variables there are
    - M: L_inf bound we want to test (non-negative)
    - property_add_constraints_fn: function(network) that re-applies input bounds and the property constraints
    Returns Marabou exit code string
    """
    # onnx_model = onnx.load(onnx_path)
    # print(onnx.helper.printable_graph(onnx_model.graph))

    network = Marabou.read_onnx(onnx_path)

    # print("\n\n\n\n=== Equations ===\n\n")
    # for eq in network.equList:
    #     typ = "EQ" if eq.EquationType == 0 else "LE" if eq.EquationType == 1 else "GE"
    #     terms = " + ".join([f"{coef}*x{var}" for (var, coef) in eq.addendList])
    #     pprint(f"{terms} {typ} {eq.scalar}")
    
    base_equation_count = len(network.equList)

    print("loaded delta model, printing inputs and output variables")
    print(network.inputVars)
    print(network.outputVars)

    # network.inputVars is list: [ [ input_vars ], [ deltas_vars ] ] depending on export.
    # Find input and deltas variables by name/position
    # Here we assume inputVars[0] -> model input, inputVars[1] -> deltas
    # If not, inspect network.inputVars to pick correct indices.
    inp_vars = network.inputVars[0].flatten()
    deltas_vars = network.inputVars[1].flatten()

    print(inp_vars)
    print(deltas_vars)
    print(counter_examples)

    # sys.exit()

    # # Input bounds: user-specific — you already set all inputs in [0,1] and input[22] in [0,0.1]
    # for i in range(len(inp_vars)):
    #     network.setLowerBound(inp_vars[i], 0.0)
    #     network.setUpperBound(inp_vars[i], 1.0)
    # # tighten the special input[22]
    # network.setLowerBound(inp_vars[22], 0.0)
    # network.setUpperBound(inp_vars[22], 0.1)

    disjuncts = []
    eps = 1e-6  # small tolerance

    for x_cex in counter_examples:
        conjunct = []
        for i, val in enumerate(x_cex):
            eq_low = Marabou.Equation()
            eq_low.addAddend(1.0, inp_vars[i])
            eq_low.setScalar(val - eps)
            eq_low.EquationType = MarabouCore.Equation.GE

            eq_high = Marabou.Equation()
            eq_high.addAddend(1.0, inp_vars[i])
            eq_high.setScalar(val + eps)
            eq_high.EquationType = MarabouCore.Equation.LE

            conjunct.extend([eq_low, eq_high])

        disjuncts.append(conjunct)

    print(disjuncts)
    network.addDisjunctionConstraint(disjuncts)
    
    # Delta bounds: -M <= delta_i <= M

    print(M)
    M = 5

    for i in range(num_deltas):
        network.setLowerBound(deltas_vars[i], -M)
        network.setUpperBound(deltas_vars[i], M)

        
    # network.setLowerBound(deltas_vars[0], -0)
    # network.setUpperBound(deltas_vars[0], 0)

    # network.setLowerBound(deltas_vars[1], 3)
    # network.setUpperBound(deltas_vars[1], 3)

    # network.setLowerBound(deltas_vars[2], 0)
    # network.setUpperBound(deltas_vars[2], 0)

    # network.setLowerBound(deltas_vars[3], 0)
    # network.setUpperBound(deltas_vars[3], 0)

    # exitCode, vals, stats = network.solve()
    # sys.exit()

    # Add the property constraints using the provided function
    property_add_constraints_fn(network)

    print("\n\n\n\n=== Equations ===\n\n")
    print(dir(network))
    print(network.numVars)
    print(network.upperBounds)
    print(network.lowerBounds)
    # sys.exit()
    for eq in network.equList:
        typ = "EQ" if eq.EquationType == 0 else "LE" if eq.EquationType == 1 else "GE"
        # terms = " + ".join([f"{coef}*x{var}" for (var, coef) in eq.addendList])
        print(eq.addendList)
        pprint(f"{eq.EquationType} {eq.scalar}")

    # print(counter_examples)
    # print("\n=== Manually Added Equations ===\n")
    # for eq in network.equList[base_equation_count:]:
    #     terms = " + ".join(
    #         [f"{coef}*x{var}" for (coef, var) in eq.addendList]
    #     )
    #     pprint(f"{terms} {eq.EquationType} {eq.scalar}")

    # sys.exit()
    print(f"Running Marabou with M = {M} ... (this may take a while)")
    
    # a = network.evaluate(inputValues = [3, 4, 0, 0, 3, 0])
    # print(a)

    options = Marabou.createOptions()
    options._solveWithMILP = True 
    options._verbosity = 2  # More detailed output
    options._timeoutInSeconds = 300  # Increase timeout

    exitCode, vals, stats = network.solve(options=options)  # adjust timeout as needed

    sys.exit()
    return exitCode, vals, stats

# -----------------------
# 4) Binary search over M
# -----------------------
def binary_search_min_M(pytorch_model, target_layer_path, counter_examples, property_add_constraints_fn,
                        M_hi=1.0, M_lo=0.0, tol=1e-3, max_iters=20):
    onnx_path, num_deltas = export_with_deltas(pytorch_model, target_layer_path, onnx_outpath="with_deltas.onnx")
    lo, hi = M_lo, M_hi
    best_M = None
    for it in range(max_iters):
        mid = (lo + hi) / 2.0
        exitCode, vals, stats = check_with_M(onnx_path, counter_examples, num_deltas, mid, property_add_constraints_fn)
        print("marabou result:", exitCode)
        if exitCode == "sat":
            # there exists modification with max magnitude mid that fixes property
            best_M = mid
            hi = mid

        elif exitCode == "unsat":
            # verifier found a counterexample -> mid is too small to fix property
            lo = mid
        else:
            # unknown or timeout: treat conservatively (increase M)
            lo = mid
        if hi - lo < tol:
            break
    return best_M, lo, hi

# -----------------------
# Example: property_add_constraints_fn for "output[1] is the smallest"
# -----------------------
def add_property_constraints_output1_smallest(network):
    outputVars = network.outputVars[0].flatten()
    eps = 1e-4  # small margin for strict inequality

    print(outputVars)
    # # Property: output[1] is the smallest among all outputs
    # for i in range(len(outputVars)):
    #     if i == 1:
    #         continue

    #     eq = Marabou.Equation(MarabouCore.Equation.LE)
    #     eq.addAddend(1.0, outputVars[1])    # + output[1]
    #     eq.addAddend(-1.0, outputVars[i])   # - output[i]
    #     eq.setScalar(-eps)                  # output[1] - output[i] <= -eps
    #     network.addEquation(eq)

    eq = Marabou.Equation(MarabouCore.Equation.GE)
    eq.addAddend(1, outputVars[0])
    eq.addAddend(-1, outputVars[1])
    eq.setScalar(0.001)
    network.addEquation(eq)

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # 1) load your PyTorch-converted onnx2pytorch model or your original PyTorch model
    # Here you already had: model = ConvertModel(onnx.load("trained_model.onnx"))
    # I recommend starting from the PyTorch version you can export reliably.
    import onnx2pytorch
    onnx_model = onnx.load("custom_network.onnx")

    print(onnx.helper.printable_graph(onnx_model.graph))
    # print(onnx_model)
    with open("data.csv") as f:
        reader = csv.reader(f)
        loaded = [[float(x) for x in row] for row in reader]

    # counter_examples = loaded
    counter_examples = [loaded[0]]
    counter_examples = [[3, 4]]

    base_model = ConvertModel(onnx_model)  # this gives a torch.nn.Module
    
    print(base_model)

    target_layer = "MatMul_output"  # replace with the actual attribute path to the linear layer in `base_model`
    best_M, lo, hi = binary_search_min_M(base_model, target_layer, counter_examples, add_property_constraints_output1_smallest,
                                        M_hi=6.0, M_lo=0.0, tol=3.0)
    print("BEST M found:", best_M, "search interval:", lo, hi)
