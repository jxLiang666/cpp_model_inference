#include "nerual_network.h"
namespace nn {
int NerualNetwork::run() {
    int ret = 0;
    ret = preprocess();
    if (ret != 0) return ret;
    ret = process();
    if (ret != 0) return ret;
    ret = postprocess();
    if (ret != 0) return ret;
    return 0;
}
}  // namespace nn
