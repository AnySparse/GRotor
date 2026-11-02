#pragma once

#include <charconv>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace grotor {

// One writer owns the FILE while a routing stage runs on another thread.
class RouteWriter {
public:
    explicit RouteWriter(FILE* file) : file_(file), buffer_(new char[capacity_]) {}

    void text(std::string_view value) {
        if (value.empty()) return;
        if (value.size() > capacity_) {
            flush();
            write(value.data(), value.size());
        } else {
            reserve(value.size());
            std::memcpy(buffer_.get() + size_, value.data(), value.size());
            size_ += value.size();
        }
    }

    void segment(int x0, int y0, int l0, int x1, int y1, int l1) {
        reserve(96);
        char* position = buffer_.get() + size_;
        auto number = [&](int value, char separator) {
            position = std::to_chars(position, buffer_.get() + capacity_, value).ptr;
            *position++ = separator;
        };
        number(x0, ' ');
        number(y0, ' ');
        number(l0, ' ');
        number(x1, ' ');
        number(y1, ' ');
        number(l1, '\n');
        size_ = position - buffer_.get();
    }

    void flush() {
        if (size_) write(buffer_.get(), size_);
        size_ = 0;
    }

private:
    void reserve(std::size_t bytes) {
        if (capacity_ - size_ < bytes) flush();
    }

    void write(const char* data, std::size_t bytes) {
        if (std::fwrite(data, 1, bytes, file_) != bytes)
            throw std::runtime_error("Failed to write routing output");
    }

    static constexpr std::size_t capacity_ = 1024 * 1024;
    FILE* file_;
    std::unique_ptr<char[]> buffer_;
    std::size_t size_ = 0;
};

} // namespace grotor
