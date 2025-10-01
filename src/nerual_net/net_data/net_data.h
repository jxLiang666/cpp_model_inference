#pragma once
#include <string>
#include <cstring>
#include <vector>
#include <iostream>
namespace nn {
struct NetData {
    explicit NetData(size_t _size) : size_(_size), data_(std::malloc(_size)) {
        // std::cout << "create NetData, size is " << _size << std::endl;
        if (!data_) {
            throw std::bad_alloc();
        }
    }
    ~NetData() {
        releaseData();
    }

    NetData(const NetData &) = delete;
    NetData &operator=(const NetData &) = delete;

    NetData(NetData &&other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
        // std::cout << "move NetData, size is " << size_ << std::endl;
    }

    NetData &operator=(NetData &&other) noexcept {
        if (this != &other) {
            releaseData();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
            // std::cout << "operator= move NetData, size is " << size_ << std::endl;
        }
        return *this;
    }

    ///< 深拷贝
    NetData copy() const {
        NetData data(size_);
        std::memcpy(data.data_, data_, size_);
        // std::cout << "copy NetData, size is " << size_ << std::endl;
        return data;
    }

    const size_t getSize() const noexcept { return size_; };
    void        *getData() const noexcept { return data_; };

    ///< 释放data
    void releaseData() noexcept {
        if (data_) {
            // std::cout << "release NetData, size is " << size_ << std::endl;
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