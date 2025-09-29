#include <fstream>
#include <string>
#include <thread>
#include <cstring>
#include <memory>
#include <opencv2/opencv.hpp>
#include "stdc.h"
#include "vis.h"
#include "net_data_op.h"
#include "net_data.h"
#include "stdc_data_adapter.h"
#include "nerual_network.h"
namespace nnt = nn::tool;

int main() {
    std::ifstream f("data/model_path.txt");
    std::string   model_path;
    std::getline(f, model_path);
    std::unique_ptr< nn::NerualNetworkBase > model = std::make_unique< nn::Stdc >(model_path);
    std::unique_ptr< nn::DataAdapterBase >   adapter = std::make_unique< nn::StdcDataAdapter >(cv::Size{1088, 1088}, cv::Size{576, 576});
    auto                                     nn = std::make_unique< nn::NerualNetwork >();
    nn->init(model, adapter);
    cv::Mat img;
    img = cv::imread("data/002_h_img.jpg");
    auto masks = nn->infer< std::vector< cv::Mat > >(img);
    for (auto &&mask : masks) {
        nnt::Vis::saveMask(mask, "mask");
    }
}