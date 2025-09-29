#pragma once
#include <memory>
#include "nerual_network_base.h"
#include "data_adapter.h"
namespace nn {
class NerualNetwork {
public:
    NerualNetwork() = default;
    NerualNetwork(NerualNetwork &&) = default;
    NerualNetwork(const NerualNetwork &) = default;
    NerualNetwork &operator=(NerualNetwork &&) = default;
    NerualNetwork &operator=(const NerualNetwork &) = default;
    virtual ~NerualNetwork() = default;

    int init(std::unique_ptr< NerualNetworkBase > &_model, std::unique_ptr< DataAdapterBase > &_adapter) {
        model_ = std::move(_model);
        adapter_ = std::move(_adapter);
        return 0;
    }

    template < typename T, typename... Args >
    auto infer(Args &&..._args) -> T {
        auto input = adapter_->createInputData(std::forward< Args >(_args)...);

        std::vector< std::vector< nn::NetData > > output;
        model_->infer(input, output);
        return adapter_->createOutputData< T >(output);
    }

protected:
    std::unique_ptr< NerualNetworkBase > model_;
    std::unique_ptr< DataAdapterBase >   adapter_;
};
}  // namespace nn