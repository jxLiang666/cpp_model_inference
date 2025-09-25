#include "onnx.h"
namespace nn {
Onnx::Onnx(const std::string &_model_path) : NerualNetwork(),
                                             env_(Ort::Env(ORT_LOGGING_LEVEL_WARNING, _model_path.c_str())),
                                             session_options_(Ort::SessionOptions(nullptr)),
                                             session_(Ort::Session(env_, model_path_.c_str(), session_options_)) {
    model_path_ = _model_path;
    name_ = _model_path;

    init();
};
int Onnx::init() {
    return 0;
}
int Onnx::deinit() {
    return 0;
}
int Onnx::infer() {
    return 0;
}
int Onnx::preprocess() {
    return 0;
}
int Onnx::process() {
    return 0;
}
Onnx::~Onnx() {
    deinit();
}
}  // namespace nn