#pragma once

#include "onnx/onnx_pb.h"
#include <map>

namespace onnx_mlir {

void replaceAll(
    std::string &str, const std::string &from, const std::string &to);

std::string legalize_name(std::string name);

struct InitializedTensorMapping {
  void AddMapping(std::string name, onnx::TensorProto tensor);

  bool ContainKey(std::string name);

private:
  std::map<std::string, onnx::TensorProto> nameToInitializedTensor;
};
} // namespace onnx_mlir