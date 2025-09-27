#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "stdc.h"
#include "vis.h"
namespace nnt = nn::tool;
int main(){
    std::ifstream f("data/model_path.txt");
    std::string model_path;
    std::getline(f,model_path);
    auto model = nn::Stdc(model_path);

    cv::Mat img;
    img = cv::imread("data/002_h_img.jpg");
    
    cv::resize(img,img,{576,576});
    std::vector<nn::NetData> input;
    input.push_back(nn::NetData(3*576*576));
    void * data = input[0].getData();
    memcpy(data,img.data,3*576*576);
    std::vector<nn::NetData> output;
    model.infer(input,output);
    nnt::Vis::saveImg(img,"h_img");
}