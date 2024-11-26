#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "onnx/onnx_pb.h"

#include <fstream>

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace onnx_mlir {

void ImportFrontendModelFile(std::string model_name, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module);
}