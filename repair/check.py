# import maraboupy.MarabouNetworkONNX
# network = maraboupy.MarabouNetworkONNX.MarabouNetworkONNX("trained_model.onnx")

# print("hello")
# print(network.inputVars)
# print(network.outputVars)

# # Get input and output variable indices
# input_vars = network.inputVars[0].flatten()  # shape (28,)
# output_vars = network.outputVars[0].flatten()  # shape (8,)

# # 1️⃣ Set bounds on inputs
# for i in range(28):
#     network.setLowerBound(input_vars[i], 0.0)
#     network.setUpperBound(input_vars[i], 1.0)

# # 2️⃣ Tighter bound for input_23 (index 22 since 0-based)
# network.setUpperBound(input_vars[22], 0.1)

# # 3️⃣ Property: Output_2 is smallest → output_2 <= output_j for all j != 2
# output_2 = output_vars[1]
# ε = 1e-3  # small margin
# for j, out in enumerate(output_vars):
#     if j != 1:
#         network.addInequality([output_2, out], [-1, 1], -ε)   # output_2 - output_j <= 0

# # 4️⃣ Solve with Marabou
# # options = Marabou.createOptions(verbosity=1)
# # vals, stats = network.solve()
# a = network.solve()
# print(a)

# # 5️⃣ Interpret result
# if vals:
#     print("❌ Property violated! Counterexample found:")
#     for i, v in enumerate(input_vars):
#         print(f"x{i+1} = {vals[v]:.4f}")
#     for i, v in enumerate(output_vars):
#         print(f"y{i+1} = {vals[v]:.4f}")
# else:
#     print("✅ Property holds for all inputs in the specified range.")