add_library(OMBuilder
    FrontendDialectTransformer.cpp
    FrontendDialectTransformer.hpp)   


target_link_libraries(OMBuilder onnx)