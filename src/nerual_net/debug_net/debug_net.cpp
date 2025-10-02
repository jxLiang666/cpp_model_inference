#include "debug_net.h"

namespace nn {
int DebugNet::init() {
    return 0;
}
int DebugNet::deinit() {
    return 0;
}
int DebugNet::infer(_IN std::vector< std::vector< NetData > > &_input, _OUT std::vector< std::vector< NetData > > &_output) {
    return 0;
}
int DebugNet::preprocess(_IN std::vector< std::vector< NetData > > &_input) {
    return 0;
}
int DebugNet::process(_OUT std::vector< std::vector< NetData > > &_output) {
    return 0;
}
int DebugNet::postprocess(_IN_OUT std::vector< std::vector< NetData > > &_output) {
    return 0;
}
}  // namespace nn