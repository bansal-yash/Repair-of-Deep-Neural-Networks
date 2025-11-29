import onnx

# After exporting, check the with_deltas.onnx file
onnx_model = onnx.load("with_deltas.onnx")
print(onnx.helper.printable_graph(onnx_model.graph))