#include "nerual_network.h"
namespace nn {
int NerualNetwork::run(std::vector< NetData > &_input, std::vector< NetData > &_output) {
    int ret = 0;
    ret = preprocess(_input);
    if (ret != 0) return ret;
    ret = process();
    if (ret != 0) return ret;
    ret = postprocess(_output);
    if (ret != 0) return ret;
    return 0;
}
}  // namespace nn
