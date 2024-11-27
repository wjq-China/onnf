#include "FrontendDialectTransformer.hpp"
#include <cstdint>

namespace onnx_mlir {
namespace {
InitializedTensorMapping initializedTensors;
class FrontendGenImpl {
public:
  FrontendGenImpl(mlir::MLIRContext &context)
      : context(context), builder_(&context) {}
  mlir::ModuleOp ImportONNXModel(onnx::ModelProto model) {
    ImportGraph(model.graph());
    return module_;
  }

private:
  mlir::MLIRContext &context;
  mlir::ModuleOp module_;
  mlir::OpBuilder builder_;
  mlir::Value onoe_;

  mlir::Location UnKnonLoc() { return mlir::UnknownLoc::get(&context); }

  mlir::Type convertONNXTypeToMLIRType(onnx::TensorProto_DataType onnxType) {
    switch (onnxType) {
    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
      return builder_.getF16Type();
    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
      return builder_.getF32Type();
    case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
      return builder_.getF64Type();
    case onnx::TensorProto_DataType::TensorProto_DataType_INT8:
    case onnx::TensorProto_DataType::TensorProto_DataType_UINT8:
      return builder_.getIntegerType(8);
    case onnx::TensorProto_DataType::TensorProto_DataType_INT16:
    case onnx::TensorProto_DataType::TensorProto_DataType_UINT16:
      return builder_.getIntegerType(16);
    case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
    case onnx::TensorProto_DataType::TensorProto_DataType_UINT32:
      return builder_.getIntegerType(32);
    case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
    case onnx::TensorProto_DataType::TensorProto_DataType_UINT64:
      return builder_.getIntegerType(64);
    case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
      return builder_.getI1Type();
    case onnx::TensorProto_DataType::TensorProto_DataType_STRING:
    case onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX64:
    case onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX128:
    case onnx::TensorProto_DataType::TensorProto_DataType_UNDEFINED:
      assert(false && "Unsupported data type encountered.");
      return nullptr;
    }
  }

  mlir::Type ImportInputTensorType(const onnx::ValueInfoProto &input) {
    std::vector<int64_t> dims;
    onnx::TensorShapeProto shape_proto = input.type().tensor_type().shape();
    auto input_tensor_legalized_name = legalize_name(input.name());

    for (int i = 0; i < shape_proto.dim_size(); i++) {
      if (shape_proto.dim()[i].dim_value()) {
        int dim_numeric_size = shape_proto.dim()[i].dim_value();
        assert(dim_numeric_size != 0 &&
               "Parsed an input tensor with a dimension size of zero");
        if (dim_numeric_size > 0) {
          dims.push_back(dim_numeric_size);
        } else {
          dims.push_back(-1);
        }
      } else {
        dims.push_back(-1); // Unspecified dimension.
      }
    }

    onnx::TensorProto_DataType elementOnnxType =
        (onnx::TensorProto_DataType)input.type().tensor_type().elem_type();
    mlir::Type element_type = convertONNXTypeToMLIRType(elementOnnxType);
    llvm::ArrayRef<int64_t> tensor_dims(dims.data(), dims.size());
    return mlir::RankedTensorType::get(tensor_dims, element_type);
  }

  void ImportGraph(
      const onnx::GraphProto &graph, const std::string &name = "main_graph") {
    for (auto initializer : graph.initializer()) {
      auto name = initializer.name();
      initializedTensors.AddMapping(legalize_name(name), initializer);
    }

    llvm::SmallVector<mlir::Type, 4> arg_types;

    for (const auto &input : graph.input()) {
      if (!initializedTensors.ContainKey(legalize_name(input.name()))) {
        arg_types.emplace_back(ImportInputTensorType(input));
      }
    }

    // Create the main function.
    auto funcType = builder_.getFunctionType(arg_types, {});
    auto mainFunc =
        mlir::FuncOp::create(builder_.getUnknownLoc(), name, funcType, {});
  }
};
} // namespace

} // namespace onnx_mlir

namespace onnx_mlir {
void ImportFrontendModelFile(std::string model_name, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module) {
  onnx::ModelProto model;
  std::ifstream input(model_name, std::ios::in | std::ios::binary);

  auto parse_success = model.ParseFromIstream(&input);
  // 得有一个东西吃这个onnx的model

  FrontendGenImpl myONNXGen(context);
  module = myONNXGen.ImportONNXModel(model);
}
} // namespace onnx_mlir