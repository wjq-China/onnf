#include <cmath>
#include <iostream>
#include <string>

#include "Builder/FrontendDialectTransformer.hpp"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"

using namespace std;
using namespace onnx_mlir;

void processInputFile(string inputFilename, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module) {
  ImportFrontendModelFile(inputFilename, context, module);
}

int main(int argc, char *argv[]) {

  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
      llvm::cl::desc("<input file>"), llvm::cl::init("-"));
  llvm::cl::ParseCommandLineOptions(argc, argv,
      "CommandLine compiler example\n\n"
      "This program blah blah blah...\n");

  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
  processInputFile(inputFilename, context, module);
  return 0;
}
