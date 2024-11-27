#include "FrontendDialectHelper.hpp"
#include <algorithm>
#include <cassert>

namespace onnx_mlir {

void replaceAll(
    std::string &str, const std::string &from, const std::string &to) {
  if (from.empty())
    return;
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length(); // In case 'to' contains 'from', like replacing
                              // 'x' with 'yx'
  }
}

std::string legalize_name(std::string name) {
  std::replace(name.begin(), name.end(), '/', '_');
  std::replace(name.begin(), name.end(), '-', '_');
  replaceAll(name, ":", "_colon_");

  if (name.size() > 0 && isdigit(name.at(0)))
    name.insert(0, 1, 'n');
  return name;
}

void InitializedTensorMapping::AddMapping(
    std::string name, onnx::TensorProto tensor) {
  assert(nameToInitializedTensor.count(name) == 0 &&
         "Tensor initializer already mapped.");
  nameToInitializedTensor.emplace(name, tensor);
}

bool InitializedTensorMapping::ContainKey(std::string name) {
  return nameToInitializedTensor.count(name) != 0;
}
} // namespace onnx_mlir