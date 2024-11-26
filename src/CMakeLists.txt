add_subdirectory(Builder)

add_executable(onnf main.cpp)

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

target_link_libraries(onnf OMBuilder)

        