import onnx

model = onnx.load("add.onnx")

graph = model.graph

print(graph.initializer)
