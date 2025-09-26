#include <fstream>
#include <string>
#include "onnx.h"
int main(){
    std::ifstream f("model_path.txt");
    std::string model_path;
    std::getline(f,model_path);
    auto model = nn::Onnx(model_path);

}