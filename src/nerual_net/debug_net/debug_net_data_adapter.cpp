#include <any>
#include <opencv2/opencv.hpp>

#include "debug_net_data_adapter.h"
#include "net_data_op.h"

namespace nn {
std::vector< std::vector< NetData > > DebugNetDataAdapter::doCreateInput(std::any &&_input) {
    std::vector< std::vector< nn::NetData > > input;
    return input;
}
std::any DebugNetDataAdapter::doCreateOutput(std::vector< std::vector< NetData > > &_output) {
    std::string s = "hello world";
    return s;
}
}  // namespace nn