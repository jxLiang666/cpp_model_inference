#pragma once
#include <string>
#include <cstring>
#include <vector>
#include <iostream>
namespace nn {
/// @brief 网络数据结构
/// @details
/// NetData 是一个简单的数据缓冲结构，用于存储神经网络输入或输出数据。
/// 它管理动态分配的内存，支持移动语义，但禁用拷贝构造和拷贝赋值，
/// 同时提供深拷贝方法 `copy()`。
struct NetData {
    explicit NetData(size_t _size) : size_(_size), data_(std::malloc(_size)) {
        std::cout << "create NetData, size is " << _size << std::endl;
        if (!data_) {
            throw std::bad_alloc();
        }
    }
    ~NetData() {
        releaseData();
    }

    NetData(const NetData &) = delete;
    NetData &operator=(const NetData &) = delete;

    NetData(NetData &&_other) noexcept
        : data_(_other.data_), size_(_other.size_) {
        _other.data_ = nullptr;
        _other.size_ = 0;
        std::cout << "move NetData, size is " << size_ << std::endl;
    }

    NetData &operator=(NetData &&_other) noexcept {
        if (this != &_other) {
            releaseData();
            data_ = _other.data_;
            size_ = _other.size_;
            _other.data_ = nullptr;
            _other.size_ = 0;
            std::cout << "operator= move NetData, size is " << size_ << std::endl;
        }
        return *this;
    }

    ///< 深拷贝
    NetData copy() const {
        NetData data(size_);
        std::memcpy(data.data_, data_, size_);
        std::cout << "copy NetData, size is " << size_ << std::endl;
        return data;
    }

    const size_t getSize() const noexcept { return size_; };
    void        *getData() const noexcept { return data_; };

    ///< 释放data
    void releaseData() noexcept {
        if (data_) {
            std::cout << "release NetData, size is " << size_ << std::endl;
            std::free(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }

private:
    size_t size_{0};
    void  *data_{nullptr};
};
}  // namespace nn