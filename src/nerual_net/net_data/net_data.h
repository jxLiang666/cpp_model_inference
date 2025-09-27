#pragma once
#include <string>
#include <vector>
namespace nn {
struct NetData {
    explicit NetData(size_t _size) : size_(_size), data_(malloc(_size)) {
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
    }

    NetData &operator=(NetData &&other) noexcept {
        if (this != &other) {
            releaseData();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    const size_t getSize() const noexcept { return size_; };
    void        *getData() const noexcept { return data_; };

private:
    void releaseData() noexcept {
        if (data_) {
            free(data_);
            data_ = nullptr;
        }
    }

private:
    size_t size_{0};
    void  *data_{nullptr};
};
}  // namespace nn