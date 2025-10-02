#pragma once
#include <vector>
#include <any>
#include "nn_config.h"
#include "net_data.h"

namespace nn {

class DataAdapterBase {
public:
    explicit DataAdapterBase() = default;
    DataAdapterBase(DataAdapterBase &&) = default;
    DataAdapterBase(const DataAdapterBase &) = default;
    DataAdapterBase &operator=(DataAdapterBase &&) = default;
    DataAdapterBase &operator=(const DataAdapterBase &) = default;
    virtual ~DataAdapterBase() = default;

    template < typename T >
    std::vector< std::vector< NetData > > createInputData(_IN T &&_input) {
        // std::cout << __PRETTY_FUNCTION__ << std::endl;
        return doCreateInput(std::make_any< std::decay_t< T > >(std::forward< T >(_input)));
    }

    template < typename T >
    T createOutputData(_OUT std::vector< std::vector< NetData > > &_output) {
        return std::any_cast< T >(this->doCreateOutput(_output));
    }

protected:
    virtual std::vector< std::vector< NetData > > doCreateInput(_IN std::any &&_input) = 0;
    virtual std::any                              doCreateOutput(_OUT std::vector< std::vector< NetData > > &_output) = 0;
};
}  // namespace nn