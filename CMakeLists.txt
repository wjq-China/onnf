cmake_minimum_required(VERSION 3.3)

project(onnf)
# 导出compile_commands.json文件，用于clangd代码补全
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 参考链接：https://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM_INCLUDE_DIRS in: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM_DEFINITIONS in: ${LLVM_DEFINITIONS}")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "MLIR_INCLUDE_DIRS in: ${MLIR_INCLUDE_DIRS}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# 将LLVM的一些宏定义转换为编译器的宏定义
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
# message(STATUS "LLVM_DEFINITIONS_LIST in: ${LLVM_DEFINITIONS_LIST}")

add_executable(onnf main.cpp
)

# target_include_directories(onnx-mlir PRIVATE ${LLVM_INCLUDE_DIRS})
# target_include_directories(onnx-mlir PRIVATE ${MLIR_INCLUDE_DIRS})

# 查找需要使用的 LLVM 组件 llvm-config --components
llvm_map_components_to_libnames(llvm_libs all)

# 链接llvm库
target_link_libraries(onnf ${llvm_libs}    
    MLIRAffineToStandard
    MLIRAffine
    MLIRLLVMIR
    MLIRStandardOps
    MLIRStandardToLLVM
    MLIRTransforms
    MLIRLoopToStandard
    MLIRVector
    MLIRLoopOps)
        