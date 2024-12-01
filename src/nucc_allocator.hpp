#ifndef CUSTOM_STL_ALLOCATOR_HPP
#define CUSTOM_STL_ALLOCATOR_HPP

#include "memory.h"
#include <cstddef>
#include <memory>
#include <vector>

namespace LAMMPS_NS {

class MemoryKeeper {
 public:
  MemoryKeeper() = delete;
  MemoryKeeper(Memory *mem) : mem(mem) {}

  ~MemoryKeeper() { clear(); }

  template <typename T> inline void store(T *&ptr, const size_t size)
  {
    infos.emplace_back(ptr, size);
  }

  void clear()
  {
    for (auto &pool : infos) { mem->destroy(pool.ptr); }
  }

 private:
  struct PoolInfo {
    template <typename T>
    PoolInfo(T *&ptr, const size_t size) :
        ptr(reinterpret_cast<void *>(ptr)), size(size * sizeof(T))
    {
    }

    PoolInfo() : ptr(nullptr), size(0){};

    void *ptr = nullptr;
    size_t size = 0;
  };

  Memory *mem;
  std::vector<PoolInfo> infos;
};

template <typename T> class CustomAllocator {
 public:
  using value_type = T;

  template <typename U> friend class CustomAllocator;

  // Constructor
  CustomAllocator(const std::size_t pool_size, Memory *const memory, MemoryKeeper *const keeper) :
      pool_size_(pool_size), memory_(memory), keeper_(keeper)
  {
  }

  // Copy constructor
  template <typename U>
  CustomAllocator(const CustomAllocator<U> &other) :
      pool_size_(other.pool_size_), memory_(other.memory_), keeper_(other.keeper_)
  {
  }

  ~CustomAllocator() {}

  // Allocate memory
  T *allocate(const std::size_t n)
  {
    T *ptr = nullptr;
    if (n > pool_size_) {
      // If requested size is larger than pool, allocate separately
      memory_->create<T>(ptr, n, "CustomAllocator_Large");
      keeper_->store(ptr, n);
      return ptr;
    }
    if (current == nullptr || left < n) {
      // Pool is full or not initialized, request a new pool
      memory_->create<T>(current, pool_size_, "CustomAllocator_Pool");
    }
    ptr = current;
    left -= n;
    current += n;
    return ptr;
  }

  // Deallocate memory
  inline void deallocate(T *p, const std::size_t n) const noexcept
  {
    // Deallocation can be handled when the allocator is destroyed
    // For pool allocator, individual deallocations are often no-ops
  }

  // Equality operators
  template <typename U> bool operator==(const CustomAllocator<U> &other) const noexcept
  {
    return (keeper_ == other.keeper_) && (memory_ == other.memory_) &&
        (pool_size_ == other.pool_size_) && (current == other.current) && (left == other.left);
  }

  template <typename U> inline bool operator!=(const CustomAllocator<U> &other) const noexcept
  {
    return !(*this == other);
  }

  size_t pool_size_;

 private:
  T *current = nullptr;
  size_t left = 0;

  MemoryKeeper *const keeper_;
  Memory *const memory_;
};

}    // namespace LAMMPS_NS

#endif    // CUSTOM_ALLOCATOR_HPP
