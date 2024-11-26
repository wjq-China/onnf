#include "FrontendDialectTransformer.hpp"
#include "onnx/onnx-ml.pb.h"

namespace onnx_mlir {
void ImportFrontendModelFile(std::string model_name, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module) {
  onnx::ModelProto model;
  std::ifstream input(model_name, std::ios::in | std::ios::binary);

  auto parse_success = model.ParseFromIstream(&input);
  // 得有一个东西吃这个onnx的model
}
} // namespace onnx_mlir