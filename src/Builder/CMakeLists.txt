add_library(OMBuilder
    FrontendDialectHelper.cpp
    FrontendDialectHelper.hpp
    FrontendDialectTransformer.cpp
    FrontendDialectTransformer.hpp
)   


target_link_libraries(OMBuilder onnx ${MLIRLibs})