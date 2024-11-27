#pragma once

#include "FrontendDialectHelper.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"
#include "onnx/onnx_pb.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <fstream>
#include <vector>

#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/StandardTypes.h" // for type convert
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace onnx_mlir {

void ImportFrontendModelFile(std::string model_name, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module);
}